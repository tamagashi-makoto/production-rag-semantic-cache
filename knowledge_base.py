"""
Expanded Knowledge Base for RAG Demo.

Contains 50+ documents across multiple categories to demonstrate
real-world RAG retrieval capabilities.
"""

# =============================================================================
# E-Commerce Knowledge Base (50+ Documents)
# =============================================================================

EXPANDED_KNOWLEDGE_BASE = {
    # =========================================================================
    # REFUND & RETURNS (10 documents)
    # =========================================================================
    "refund_policy_general": {
        "title": "General Refund Policy",
        "category": "returns",
        "content": """Our standard refund policy allows customers to return most items within 30 days 
of purchase for a full refund. Items must be unused, in original packaging, and accompanied by 
the original receipt or order confirmation. Refunds are processed within 5-7 business days after 
we receive and inspect the returned item. The refund will be issued to the original payment method."""
    },
    "refund_electronics": {
        "title": "Electronics Return Policy",
        "category": "returns",
        "content": """Electronics can be returned within 15 days of purchase. All electronics must 
include original accessories, manuals, and packaging. Opened software, video games, and digital 
downloads are non-returnable. TVs over 55 inches require a restocking fee of 15%. Defective 
electronics can be exchanged within 30 days."""
    },
    "refund_clothing": {
        "title": "Clothing Return Policy",
        "category": "returns",
        "content": """Clothing items may be returned within 60 days of purchase. Items must be 
unworn, unwashed, and have all original tags attached. Swimwear and underwear must have hygiene 
liners intact. Final sale items are marked and cannot be returned. Free return shipping is 
available for exchanges."""
    },
    "refund_international": {
        "title": "International Returns",
        "category": "returns",
        "content": """International customers can return items within 45 days. Return shipping 
costs are the responsibility of the customer. Items must be shipped with tracking. Refunds 
exclude original shipping charges. Customs duties and taxes are non-refundable. We recommend 
using our prepaid return label service for $15."""
    },
    "exchange_process": {
        "title": "Exchange Process",
        "category": "returns",
        "content": """To exchange an item, initiate a return through your account and place a 
new order for the desired item. We offer free standard shipping on exchange orders. Size 
exchanges for clothing are free. For faster processing, use our in-store exchange option 
at any retail location."""
    },
    "refund_payment_methods": {
        "title": "Refund Payment Methods",
        "category": "returns",
        "content": """Credit card refunds appear within 5-10 business days depending on your bank. 
PayPal refunds process within 24-48 hours. Store credit is issued immediately. Gift card 
purchases are refunded to a new gift card. Cash refunds are only available for in-store 
purchases with receipt."""
    },
    "damaged_items": {
        "title": "Damaged or Defective Items",
        "category": "returns",
        "content": """If you receive a damaged or defective item, contact us within 48 hours with 
photos. We will send a prepaid return label and priority ship a replacement. No restocking 
fees apply. If the item is out of stock, we offer a full refund plus a 10% courtesy credit."""
    },
    "return_shipping_labels": {
        "title": "Return Shipping Labels",
        "category": "returns",
        "content": """Free return labels are available for orders over $50. For orders under $50, 
a $7.95 return shipping fee is deducted from your refund. Premium members always receive free 
return shipping. Labels expire after 30 days. You can print labels from your order history."""
    },
    "gift_returns": {
        "title": "Gift Returns",
        "category": "returns",
        "content": """Gift recipients can return items for store credit without a receipt. If 
you have the order number or gift receipt, we can process a refund to the original purchaser. 
Gift returns must be made within 90 days of the original purchase date."""
    },
    "holiday_returns": {
        "title": "Holiday Return Policy",
        "category": "returns",
        "content": """Extended holiday returns apply to purchases made November 1 through December 31. 
These items can be returned until January 31 of the following year. All other return policy 
conditions still apply. This extension does not apply to electronics or final sale items."""
    },

    # =========================================================================
    # SHIPPING & DELIVERY (10 documents)
    # =========================================================================
    "shipping_standard": {
        "title": "Standard Shipping",
        "category": "shipping",
        "content": """Standard shipping takes 5-7 business days and costs $5.95 for orders under $50. 
Orders over $50 qualify for free standard shipping. Processing time is 1-2 business days. 
Tracking is provided via email once shipped. Delivery times may vary for rural areas."""
    },
    "shipping_express": {
        "title": "Express Shipping",
        "category": "shipping",
        "content": """Express shipping delivers in 2-3 business days for $12.95. Order by 2 PM EST 
for same-day processing. Available to all 48 contiguous US states. Express orders include 
signature confirmation. Weekend delivery is available for an additional $5."""
    },
    "shipping_overnight": {
        "title": "Overnight Shipping",
        "category": "shipping",
        "content": """Overnight shipping guarantees next-business-day delivery for $24.95. Orders 
must be placed by 1 PM EST. Available Monday through Friday only. PO Boxes are not eligible. 
Saturday delivery costs an additional $10. Not available to Alaska or Hawaii."""
    },
    "shipping_international": {
        "title": "International Shipping",
        "category": "shipping",
        "content": """We ship to over 100 countries worldwide. International shipping starts at $19.95 
and takes 7-21 business days depending on destination. Customs duties and taxes are the 
responsibility of the recipient. DHL and FedEx international tracking is provided."""
    },
    "shipping_tracking": {
        "title": "Order Tracking",
        "category": "shipping",
        "content": """Track your order through your account or the link in your shipping confirmation 
email. Updates appear within 24 hours of shipping. Real-time GPS tracking is available for 
express and overnight orders. Contact support if tracking hasn't updated in 3 days."""
    },
    "delivery_instructions": {
        "title": "Delivery Instructions",
        "category": "shipping",
        "content": """Add delivery instructions during checkout or from your account settings. 
Options include: leave at door, require signature, hold at carrier facility, or deliver to 
neighbor. Changes to delivery instructions must be made before the order ships."""
    },
    "shipping_restrictions": {
        "title": "Shipping Restrictions",
        "category": "shipping",
        "content": """Some items cannot ship to all addresses due to carrier restrictions or local 
regulations. Hazardous materials, oversized items, and certain electronics have shipping 
limitations. APO/FPO addresses may have additional restrictions and longer transit times."""
    },
    "shipping_delays": {
        "title": "Shipping Delays",
        "category": "shipping",
        "content": """During peak seasons (holidays, major sales), shipping times may be extended 
by 2-3 days. Weather events can cause delays. We will notify you of significant delays. 
Guaranteed delivery dates are extended accordingly. No shipping fee refunds for weather delays."""
    },
    "local_pickup": {
        "title": "Store Pickup",
        "category": "shipping",
        "content": """Buy online, pick up in store (BOPIS) is free and usually ready within 2 hours. 
Same-day pickup cutoff is 4 PM. Items are held for 7 days. Bring your order confirmation and 
valid ID. Curbside pickup is available at select locations."""
    },
    "shipping_large_items": {
        "title": "Large Item Shipping",
        "category": "shipping",
        "content": """Furniture and large appliances require freight shipping. Delivery takes 1-3 
weeks and includes in-home delivery. White glove service with assembly is available for $99. 
Appointments are scheduled in 4-hour windows. Basement and upstairs delivery may incur fees."""
    },

    # =========================================================================
    # WARRANTY & PROTECTION (10 documents)
    # =========================================================================
    "warranty_standard": {
        "title": "Standard Warranty",
        "category": "warranty",
        "content": """All products include a 1-year manufacturer warranty covering defects in 
materials and workmanship. Normal wear and tear is not covered. Water damage voids warranty 
unless the product is water-resistant rated. Keep your receipt for warranty claims."""
    },
    "warranty_extended": {
        "title": "Extended Warranty Options",
        "category": "warranty",
        "content": """Extended protection plans are available for 2 or 3 years beyond the 
manufacturer warranty. Plans cover mechanical failures, power surge damage, and accidental 
damage (with AccidentalDamage+ plan). Cancel within 30 days for a full refund."""
    },
    "warranty_electronics": {
        "title": "Electronics Warranty",
        "category": "warranty",
        "content": """Electronics are covered by manufacturer warranty for 1 year. Screens, 
batteries, and accessories may have shorter coverage periods. Register your product online 
for easy warranty claims. Software issues are not covered under warranty."""
    },
    "warranty_appliances": {
        "title": "Appliance Warranty",
        "category": "warranty",
        "content": """Major appliances carry a 2-year warranty on parts and labor. Compressors 
and motors may have extended coverage up to 10 years. Installation must be done by certified 
technicians to maintain warranty. Commercial use voids residential warranties."""
    },
    "warranty_claims": {
        "title": "Filing Warranty Claims",
        "category": "warranty",
        "content": """File warranty claims online or call our support line. Have your receipt and 
product serial number ready. Claims are processed within 3-5 business days. We offer repair, 
replacement, or refund based on product availability and condition."""
    },
    "protection_plan_details": {
        "title": "Protection Plan Coverage Details",
        "category": "warranty",
        "content": """Protection plans cover: mechanical and electrical failures, normal wear 
failures, manufacturer defects after warranty expires. AccidentalDamage+ adds: drops, spills, 
cracks, and electrical surges. No deductibles on most claims. $0 limit on claims."""
    },
    "warranty_exclusions": {
        "title": "Warranty Exclusions",
        "category": "warranty",
        "content": """Warranties do not cover: cosmetic damage, unauthorized modifications, 
commercial use of consumer products, theft or loss, damage from disasters, pre-existing 
conditions, and consumable parts like batteries after 90 days."""
    },
    "appliance_service": {
        "title": "Appliance Service Appointments",
        "category": "warranty",
        "content": """Schedule in-home service for appliances through your account or by calling 
support. Technicians are available Monday-Saturday. Same-week appointments usually available. 
A $99 diagnostic fee applies but is waived if repairs are completed."""
    },
    "product_registration": {
        "title": "Product Registration",
        "category": "warranty",
        "content": """Register your products online within 30 days of purchase to activate warranty 
and receive safety notices. Registration is required for extended warranty activation. 
Include serial number, purchase date, and receipt scan."""
    },
    "price_protection": {
        "title": "Price Protection Guarantee",
        "category": "warranty",
        "content": """If an item goes on sale within 14 days of purchase, we'll refund the 
difference. Price protection applies to identical items only. Excludes clearance, lightning 
deals, and third-party marketplace items. Submit requests through customer service."""
    },

    # =========================================================================
    # CUSTOMER SUPPORT (10 documents)
    # =========================================================================
    "support_contact": {
        "title": "Contact Customer Support",
        "category": "support",
        "content": """Reach our support team via: Live Chat (24/7, fastest response), Phone 
1-800-EXAMPLE (Mon-Sat 8AM-10PM EST), Email support@example.com (24-48 hour response), 
Twitter @ExampleSupport. Premium members have dedicated priority support lines."""
    },
    "support_chat": {
        "title": "Live Chat Support",
        "category": "support",
        "content": """Live chat is available 24/7 on our website and mobile app. Average wait 
time is under 2 minutes. Chat transcripts are emailed for your records. You can share 
screenshots and files directly in chat. Chat bot can handle simple inquiries instantly."""
    },
    "support_phone": {
        "title": "Phone Support",
        "category": "support",
        "content": """Call 1-800-EXAMPLE for phone support. Hours: Monday-Saturday 8AM-10PM EST, 
Sunday 10AM-6PM EST. Have your order number ready. Callback option available during high 
volume. Spanish language support available. TTY service at 1-800-EXAMPLE-TTY."""
    },
    "support_email": {
        "title": "Email Support",
        "category": "support",
        "content": """Email support@example.com for non-urgent inquiries. Include your order 
number in the subject line. Response time is 24-48 hours. Attach photos for damage claims. 
Do not send sensitive information like full credit card numbers via email."""
    },
    "support_social": {
        "title": "Social Media Support",
        "category": "support",
        "content": """Reach us on Twitter @ExampleSupport or Facebook Messenger. Response time 
is typically 1-4 hours during business hours. Direct messages only for order-specific issues. 
Do not share personal information in public posts."""
    },
    "support_self_service": {
        "title": "Self-Service Options",
        "category": "support",
        "content": """Handle many issues yourself through your account: track orders, print 
return labels, update payment methods, change addresses, cancel orders (within 30 minutes), 
and manage subscriptions. Our Help Center has over 500 articles and video tutorials."""
    },
    "support_escalation": {
        "title": "Escalating Issues",
        "category": "support",
        "content": """If your issue isn't resolved, ask to speak with a supervisor. You can 
also email escalations@example.com for executive review. Include your case number and 
previous communication summary. Escalated cases are reviewed within 24 hours."""
    },
    "support_accessibility": {
        "title": "Accessibility Support",
        "category": "support",
        "content": """We are committed to accessibility. Our website follows WCAG 2.1 guidelines. 
Screen reader optimized pages available. Large print materials on request. Sign language 
video support available by appointment. Contact accessibility@example.com for assistance."""
    },
    "feedback_complaints": {
        "title": "Feedback and Complaints",
        "category": "support",
        "content": """Share feedback at feedback.example.com. Complaints are logged and reviewed 
by our quality team. We aim to resolve complaints within 5 business days. Compliments are 
shared with team members. Survey invitations sent after support interactions."""
    },
    "support_hours_holidays": {
        "title": "Holiday Support Hours",
        "category": "support",
        "content": """Modified holiday hours: Thanksgiving - closed, Black Friday - 6AM-12AM, 
Christmas Eve - 8AM-5PM, Christmas Day - closed, New Year's Eve - 8AM-6PM. Chat support 
remains available with limited staffing. Email responses may be delayed 48-72 hours."""
    },

    # =========================================================================
    # PRODUCTS & CATEGORIES (10 documents)
    # =========================================================================
    "product_electronics_tv": {
        "title": "Television Buying Guide",
        "category": "products",
        "content": """Choose the right TV size based on viewing distance: 9-14 feet = 65-75 inch, 
6-9 feet = 50-55 inch, 4-6 feet = 32-43 inch. OLED offers best picture quality, QLED best for 
bright rooms. 4K is standard, 8K for future-proofing. Smart TV features include streaming apps 
and voice control. Refresh rate of 120Hz is ideal for gaming."""
    },
    "product_electronics_laptop": {
        "title": "Laptop Buying Guide",
        "category": "products",
        "content": """Consider use case: Basic (4GB RAM, 128GB) for browsing and documents, 
Mainstream (8GB RAM, 256GB SSD) for multitasking, Performance (16GB+ RAM, 512GB+ SSD) for 
creative work and gaming. Battery life ranges from 6-15 hours. Screen size 13-14 inch for 
portability, 15-17 inch for productivity. MacOS, Windows, or ChromeOS based on preference."""
    },
    "product_appliances_refrigerator": {
        "title": "Refrigerator Buying Guide",
        "category": "products",
        "content": """Measure your space carefully including door swing clearance. French door models 
offer premium features and wide shelves. Side-by-side provides equal fridge/freezer access. 
Top freezer is most affordable. Counter-depth for seamless look. Smart refrigerators offer 
inventory tracking and temperature alerts. Energy Star models save 15% on electricity."""
    },
    "product_appliances_washer": {
        "title": "Washing Machine Guide",
        "category": "products",
        "content": """Front-load washers are most efficient and gentle on clothes. Top-load is 
faster and more affordable. Capacity: 4.5 cu ft for households of 3-4, 5.0+ for larger families. 
Steam cleaning removes allergens. WiFi connectivity allows remote monitoring. High spin speeds 
reduce drying time. All-in-one washer-dryers save space in apartments."""
    },
    "product_furniture_sofa": {
        "title": "Sofa Buying Guide",
        "category": "products",
        "content": """Measure doorways and hallways for delivery access. Standard sofa is 84-90 
inches, loveseat is 52-63 inches. Sectionals offer flexible configurations. Leather is durable 
and easy to clean, fabric offers more color options. Frame should be kiln-dried hardwood. 
High-resilience foam cushions last longer. Warranty should cover frame and cushions separately."""
    },
    "product_outdoor_grill": {
        "title": "Grill Buying Guide",
        "category": "products",
        "content": """Gas grills offer convenience and temperature control. Charcoal provides 
smoky flavor. Pellet grills combine both benefits. BTU per square inch (80-100) matters more 
than total BTU. Stainless steel burners last longest. Multiple cooking zones add versatility. 
Built-in grills for outdoor kitchens. Portable grills for tailgating. Consider propane vs 
natural gas connection."""
    },
    "product_fitness_treadmill": {
        "title": "Treadmill Buying Guide",
        "category": "products",
        "content": """Motor: 2.5 HP for walking, 3.0+ HP for running. Belt width 20-22 inches 
for comfortable stride. Incline range 0-15% for varied workouts. Deck cushioning reduces 
joint impact. Built-in workouts and app connectivity for motivation. Folding design saves 
space. Check weight capacity and warranty (motor often lifetime, parts 2-5 years)."""
    },
    "product_baby_carseat": {
        "title": "Car Seat Safety Guide",
        "category": "products",
        "content": """Infant seats (rear-facing) for newborns to 35 lbs. Convertible seats 
transition rear to forward-facing (up to 65 lbs). Booster seats for children 40-100 lbs. 
All seats must meet FMVSS 213 standards. Check expiration dates (6-10 years). LATCH system 
for easy installation. Never use a car seat after an accident. Free installation checks at 
local fire stations."""
    },
    "product_kitchen_mixer": {
        "title": "Stand Mixer Guide",
        "category": "products",
        "content": """Tilt-head mixers are better for smaller batches and easy bowl access. 
Bowl-lift design for larger capacity and heavy doughs. 5-quart bowl handles most home baking. 
Wattage of 300-500 for standard use, 575+ for bread kneading. Attachments expand functionality 
to pasta making, meat grinding, and food processing. Metal construction is most durable."""
    },
    "product_gaming_console": {
        "title": "Gaming Console Guide",
        "category": "products",
        "content": """PlayStation 5: Best exclusive games, powerful hardware, VR capability. 
Xbox Series X: Game Pass subscription value, backward compatibility. Nintendo Switch: 
Portable gaming, family-friendly exclusives. Consider: exclusive game titles, online 
subscription costs, storage capacity (expandable options), controller preferences. All 
support 4K gaming and streaming apps."""
    },

    # =========================================================================
    # PAYMENT & BILLING (10+ documents)
    # =========================================================================
    "payment_methods": {
        "title": "Accepted Payment Methods",
        "category": "payment",
        "content": """We accept: Visa, Mastercard, American Express, Discover, PayPal, Apple Pay, 
Google Pay, Shop Pay, Affirm financing, and store gift cards. Prepaid cards and debit cards 
are accepted. Cryptocurrency not currently accepted. International cards with Visa/MC logos 
work but may incur foreign transaction fees."""
    },
    "payment_financing": {
        "title": "Financing Options",
        "category": "payment",
        "content": """Affirm financing available at checkout for purchases over $100. Options 
include 3, 6, or 12 month plans. 0% APR available for qualified buyers. Quick approval 
process with soft credit check. Pay off early with no penalties. Late payments may affect 
credit score. Store credit card offers 5% back on all purchases."""
    },
    "payment_gift_cards": {
        "title": "Gift Cards",
        "category": "payment",
        "content": """Physical and e-gift cards available in $25-$500 denominations. Custom 
amounts available online. E-gift cards delivered instantly via email. Physical cards ship 
free. Gift cards never expire and have no fees. Lost cards can be replaced with proof of 
purchase. Bulk gift card orders for businesses available."""
    },
    "billing_invoices": {
        "title": "Invoices and Receipts",
        "category": "payment",
        "content": """Order receipts are emailed automatically after purchase. Access and print 
invoices from your order history. Business accounts can request itemized invoices. VAT 
invoices available for international orders. Receipts are retained for 7 years in your 
account. Contact support for copies of older receipts."""
    },
    "billing_disputes": {
        "title": "Billing Disputes",
        "category": "payment",
        "content": """If you notice an incorrect charge, contact us within 60 days. Have your 
order number and statement ready. Disputes are investigated within 5 business days. During 
investigation, hold is placed on the charge. Resolution options include refund, credit, or 
explanation of charge. We work with your card issuer to resolve disputes."""
    },
    "promo_codes": {
        "title": "Promo Codes and Discounts",
        "category": "payment",
        "content": """Enter promo codes at checkout in the 'Discount Code' field. Only one 
promo code per order. Promo codes cannot be combined with other offers unless specified. 
Check code terms for minimum purchase and exclusions. Sign up for emails to receive 
exclusive discount codes. Birthday discounts for account holders."""
    },
    "rewards_program": {
        "title": "Rewards Program",
        "category": "payment",
        "content": """Earn 1 point per dollar spent. 100 points = $5 reward. Points expire 
after 12 months of inactivity. Bonus points during promotional periods. Double points on 
your birthday month. Redeem points at checkout or save for larger rewards. Check your 
balance in the mobile app or online account."""
    },
}


def get_category_documents(category: str) -> dict:
    """Get all documents in a specific category."""
    return {
        doc_id: doc 
        for doc_id, doc in EXPANDED_KNOWLEDGE_BASE.items() 
        if doc.get("category") == category
    }


def search_knowledge_base(query: str) -> list:
    """Simple keyword search for filtering documents."""
    query_lower = query.lower()
    results = []
    for doc_id, doc in EXPANDED_KNOWLEDGE_BASE.items():
        if query_lower in doc["content"].lower() or query_lower in doc["title"].lower():
            results.append((doc_id, doc))
    return results


# Summary statistics
KNOWLEDGE_BASE_STATS = {
    "total_documents": len(EXPANDED_KNOWLEDGE_BASE),
    "categories": {
        "returns": len(get_category_documents("returns")),
        "shipping": len(get_category_documents("shipping")),
        "warranty": len(get_category_documents("warranty")),
        "support": len(get_category_documents("support")),
        "products": len(get_category_documents("products")),
        "payment": len(get_category_documents("payment")),
    },
    "total_content_chars": sum(len(doc["content"]) for doc in EXPANDED_KNOWLEDGE_BASE.values()),
}
