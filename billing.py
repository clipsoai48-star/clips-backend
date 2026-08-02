"""
Square billing integration.

FLOW
1. A logged-in user hits POST /billing/create-checkout-link.
2. We ask Square for a hosted checkout page tied to your Pro subscription
   plan, and return its URL.
3. The frontend redirects the browser there. The user enters card details on
   Square's own page — you never touch card numbers.
4. Once they subscribe, Square calls POST /billing/webhook/square. We verify
   the request really came from Square, look up which of your users just
   paid (by email), and flip their is_paid_tier to True.

ONE-TIME SETUP (done by whoever owns the Square account, in the dashboard —
not in this code):
  1. Square Developer Dashboard -> your app -> copy the Access Token and
     Location ID.
  2. Square Dashboard -> Items -> Subscriptions -> create a "Pro" plan at
     $12/month -> copy its Plan Variation ID.
  3. Square Developer Dashboard -> Webhooks -> Add Endpoint:
       https://<your-backend-domain>/billing/webhook/square
     Subscribe to: subscription.created, subscription.updated
     -> copy the Webhook Signature Key.

Add all of these to your .env (see the list at the bottom of this file).
"""
import os
import hmac
import hashlib
import base64
import logging

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from database import get_db
from models_db import User
from auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/billing", tags=["billing"])

SQUARE_ACCESS_TOKEN = os.environ["SQUARE_ACCESS_TOKEN"]
SQUARE_LOCATION_ID = os.environ["SQUARE_LOCATION_ID"]
SQUARE_ENVIRONMENT = os.environ.get("SQUARE_ENVIRONMENT", "sandbox")  # "sandbox" or "production"
SQUARE_PLAN_VARIATION_ID = os.environ["SQUARE_SUBSCRIPTION_PLAN_VARIATION_ID"]
SQUARE_ITEM_VARIATION_ID = os.environ["SQUARE_ITEM_VARIATION_ID"]
SQUARE_WEBHOOK_SIGNATURE_KEY = os.environ["SQUARE_WEBHOOK_SIGNATURE_KEY"]
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

SQUARE_API_BASE = (
    "https://connect.squareup.com"
    if SQUARE_ENVIRONMENT == "production"
    else "https://connect.squareupsandbox.com"
)

SQUARE_HEADERS = {
    "Square-Version": "2024-10-17",
    "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}


@router.post("/create-checkout-link")
def create_checkout_link(current_user: User = Depends(get_current_user)):
    """Returns a Square-hosted checkout URL for the logged-in user to subscribe to Pro."""
    body = {
        "idempotency_key": f"checkout-{current_user.id}-{os.urandom(8).hex()}",
        "order": {
            "location_id": SQUARE_LOCATION_ID,
            "line_items": [
                {
                    "quantity": "1",
                    "catalog_object_id": SQUARE_ITEM_VARIATION_ID,
                }
            ],
        },
        "checkout_options": {
            "subscription_plan_id": SQUARE_PLAN_VARIATION_ID,
            "redirect_url": f"{FRONTEND_URL}/billing/success",
        },
        "pre_populated_data": {
            "buyer_email": current_user.email,
        },
    }
    resp = requests.post(
        f"{SQUARE_API_BASE}/v2/online-checkout/payment-links",
        headers=SQUARE_HEADERS,
        json=body,
        timeout=10,
    )
    if resp.status_code != 200:
        logger.error("Square checkout link creation failed: %s", resp.text)
        raise HTTPException(status_code=502, detail="Could not create checkout link")

    payment_link = resp.json()["payment_link"]
    return {"checkout_url": payment_link["url"]}


def _verify_square_signature(raw_body: bytes, signature_header: str, notification_url: str) -> bool:
    combined = notification_url + raw_body.decode("utf-8")
    computed = base64.b64encode(
        hmac.new(
            SQUARE_WEBHOOK_SIGNATURE_KEY.encode("utf-8"),
            combined.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    return hmac.compare_digest(computed, signature_header)


@router.post("/webhook/square")
async def square_webhook(request: Request, db: Session = Depends(get_db)):
    raw_body = await request.body()
    signature = request.headers.get("x-square-hmacsha256-signature", "")
    notification_url = str(request.url)

    if not _verify_square_signature(raw_body, signature, notification_url):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = await request.json()
    event_type = event.get("type", "")
    logger.info("Square webhook received: %s", event_type)

    if event_type in ("subscription.created", "subscription.updated"):
        subscription = event.get("data", {}).get("object", {}).get("subscription", {})
        square_customer_id = subscription.get("customer_id")
        square_subscription_id = subscription.get("id")
        sub_status = subscription.get("status")  # "ACTIVE", "CANCELED", "PAUSED", etc.

        if not square_customer_id:
            return {"received": True}

        # Square generated the customer during checkout, so we match them
        # back to one of our users by email rather than by an ID we'd have
        # had to create ahead of time.
        customer_resp = requests.get(
            f"{SQUARE_API_BASE}/v2/customers/{square_customer_id}",
            headers=SQUARE_HEADERS,
            timeout=10,
        )
        buyer_email = None
        if customer_resp.status_code == 200:
            buyer_email = customer_resp.json().get("customer", {}).get("email_address")

        user = db.query(User).filter(User.email == buyer_email).first() if buyer_email else None

        if user:
            user.square_customer_id = square_customer_id
            user.square_subscription_id = square_subscription_id
            user.is_paid_tier = sub_status == "ACTIVE"
            db.commit()
            logger.info("Updated user %s: is_paid_tier=%s", user.id, user.is_paid_tier)
        else:
            logger.warning(
                "Square webhook: no matching user for customer %s (email=%s)",
                square_customer_id, buyer_email,
            )

    return {"received": True}


# ---------------------------------------------------------------------------
# .env variables this file needs — add these to your backend's .env:
#
# SQUARE_ACCESS_TOKEN=your-access-token
# SQUARE_LOCATION_ID=your-location-id
# SQUARE_ENVIRONMENT=sandbox        # switch to "production" when you go live
# SQUARE_SUBSCRIPTION_PLAN_VARIATION_ID=your-plan-variation-id
# SQUARE_WEBHOOK_SIGNATURE_KEY=your-webhook-signature-key
# FRONTEND_URL=https://frontend-five-sooty-02iy2icjz8.vercel.app
# ---------------------------------------------------------------------------
