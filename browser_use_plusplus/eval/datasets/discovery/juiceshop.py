from browser_use_plusplus.eval.datasets.base import Route

JUICE_SHOP_HOMEPAGE = {
    "path": "/",
    "routes": [
        # Route(method="GET", template="/rest/products/search?q="),
        # Route(method="GET", template="/api/Quantitys/"),
        # Route(method="GET", template="/rest/basket/:id"),
        # Route(method="GET", template="/api/BasketItems/:id"),
        # Route(method="PUT", template="/api/BasketItems/:id"),
        Route(method="POST", template="/api/BasketItems/"),
        # Route(method="GET", template="/api/Products/:id"),
        Route(method="GET", template="/rest/products/:id/reviews"),
        Route(method="PUT", template="/rest/products/:id/reviews"),
        # Route(method="PATCH", template="/rest/products/reviews"),
        Route(method="POST", template="/rest/products/reviews"),
    ]
}

JUICE_SHOP_ADMINISTRATION = {
    "path": "/administration",
    "routes": [
        Route(method="GET", template="/rest/user/authentication-details/"),
        Route(method="GET", template="/api/Feedbacks/"),
        Route(method="DELETE", template="/api/Feedbacks/:id"),
        Route(method="GET", template="/api/Users/:id"),
    ]
}

JUICE_SHOP_ACCOUNTING = {
    "path": "/accounting",
    "routes": [
        Route(method="GET", template="/api/Quantitys/"),
        Route(method="GET", template="/rest/products/search?q="),
        Route(method="GET", template="/rest/order-history/orders"),
        Route(method="PUT", template="/api/Quantitys/:id"),
        Route(method="PUT", template="/api/Products/:id"),
        Route(method="PUT", template="/rest/order-history/:id/delivery-status"),
    ]
}

JUICE_SHOP_ABOUT = {
    "path": "/about",
    "routes": [
        Route(method="GET", template="/rest/admin/application-configuration"),
        Route(method="GET", template="/api/Feedbacks/"),
    ]
}

JUICE_SHOP_ADDRESS_SELECT = {
    "path": "/address/select",
    "routes": [
        Route(method="GET", template="/api/Addresss/"),
        Route(method="DELETE", template="/api/Addresss/:id"),
    ]
}

JUICE_SHOP_ADDRESS_SAVED = {
    "path": "/address/saved",
    "routes": [
        Route(method="GET", template="/api/Addresss/"),
        Route(method="DELETE", template="/api/Addresss/:id"),
    ]
}

JUICE_SHOP_ADDRESS_CREATE = {
    "path": "/address/create",
    "routes": [
        Route(method="POST", template="/api/Addresss/"),
    ]
}

JUICE_SHOP_ADDRESS_EDIT = {
    "path": "/address/edit/:addressId",
    "routes": [
        Route(method="GET", template="/api/Addresss/:id"),
        Route(method="PUT", template="/api/Addresss/:id"),
    ]
}

JUICE_SHOP_DELIVERY_METHOD = {
    "path": "/delivery-method",
    "routes": [
        Route(method="GET", template="/api/Addresss/:id"),
        Route(method="GET", template="/api/Deliverys"),
    ]
}

JUICE_SHOP_DELUXE_MEMBERSHIP = {
    "path": "/deluxe-membership",
    "routes": [
        Route(method="GET", template="/rest/deluxe-membership"),
        Route(method="GET", template="/rest/admin/application-configuration"),
    ]
}

JUICE_SHOP_SAVED_PAYMENT_METHODS = {
    "path": "/saved-payment-methods",
    "routes": [
        Route(method="GET", template="/api/Cards"),
        Route(method="POST", template="/api/Cards/"),
        Route(method="DELETE", template="/api/Cards/:id"),
    ]
}

JUICE_SHOP_BASKET = {
    "path": "/basket",
    "routes": [
        Route(method="GET", template="/rest/basket/:id"),
        Route(method="GET", template="/api/BasketItems/:id"),
        Route(method="PUT", template="/api/BasketItems/:id"),
        Route(method="DELETE", template="/api/BasketItems/:id"),
    ]
}

JUICE_SHOP_ORDER_COMPLETION = {
    "path": "/order-completion/:id",
    "routes": [
        Route(method="GET", template="/rest/track-order/:id"),
        Route(method="GET", template="/api/Addresss/:id"),
    ]
}

JUICE_SHOP_CONTACT = {
    "path": "/contact",
    "routes": [
        Route(method="GET", template="/rest/user/whoami"),
        Route(method="GET", template="/rest/captcha/"),
        Route(method="POST", template="/api/Feedbacks/"),
    ]
}

JUICE_SHOP_PHOTO_WALL = {
    "path": "/photo-wall",
    "routes": [
        Route(method="GET", template="/rest/memories/"),
        Route(method="POST", template="/rest/memories"),
    ]
}

JUICE_SHOP_COMPLAIN = {
    "path": "/complain",
    "routes": [
        Route(method="POST", template="/file-upload"),
        Route(method="GET", template="/rest/user/whoami"),
        Route(method="POST", template="/api/Complaints/"),
    ]
}

JUICE_SHOP_ORDER_SUMMARY = {
    "path": "/order-summary",
    "routes": [
        Route(method="GET", template="/api/Deliverys/:id"),
        Route(method="GET", template="/api/Addresss/:id"),
        Route(method="GET", template="/api/Cards/:id"),
        Route(method="POST", template="/rest/basket/:id/checkout"),
    ]
}

JUICE_SHOP_ORDER_HISTORY = {
    "path": "/order-history",
    "routes": [
        Route(method="GET", template="/rest/order-history"),
        Route(method="GET", template="/api/Products/:id"),
    ]
}

JUICE_SHOP_PAYMENT = {
    "path": "/payment/:entity",
    "routes": [
        Route(method="GET", template="/rest/wallet/balance"),
        Route(method="GET", template="/rest/deluxe-membership"),
        Route(method="GET", template="/api/Deliverys/:id"),
        Route(method="PUT", template="/rest/basket/:id/coupon/:coupon"),
        Route(method="POST", template="/rest/deluxe-membership"),
        Route(method="PUT", template="/rest/wallet/balance"),
    ]
}

JUICE_SHOP_WALLET = {
    "path": "/wallet",
    "routes": [
        Route(method="GET", template="/rest/wallet/balance"),
    ]
}

JUICE_SHOP_LOGIN = {
    "path": "/login",
    "routes": [
        Route(method="GET", template="/rest/admin/application-configuration"),
        Route(method="POST", template="/rest/user/login"),
        Route(method="GET", template="/rest/basket/:id"),
    ]
}

JUICE_SHOP_FORGOT_PASSWORD = {
    "path": "/forgot-password",
    "routes": [
        Route(method="GET", template="/rest/user/security-question?email="),
        Route(method="POST", template="/rest/user/reset-password"),
    ]
}

JUICE_SHOP_RECYCLE = {
    "path": "/recycle",
    "routes": [
        Route(method="GET", template="/rest/admin/application-configuration"),
        Route(method="GET", template="/rest/user/whoami"),
        Route(method="GET", template="/api/Recycles/"),
        Route(method="POST", template="/api/Recycles/"),
    ]
}

JUICE_SHOP_REGISTER = {
    "path": "/register",
    "routes": [
        Route(method="GET", template="/api/SecurityQuestions/"),
        Route(method="POST", template="/api/Users/"),
        Route(method="POST", template="/api/SecurityAnswers/"),
    ]
}

JUICE_SHOP_SEARCH = {
    "path": "/search",
    "routes": [
        Route(method="GET", template="/rest/products/search?q="),
        Route(method="GET", template="/api/Quantitys/"),
        Route(method="GET", template="/rest/basket/:id"),
        Route(method="GET", template="/api/BasketItems/:id"),
        Route(method="PUT", template="/api/BasketItems/:id"),
        Route(method="POST", template="/api/BasketItems/"),
        Route(method="GET", template="/api/Products/:id"),
        Route(method="GET", template="/rest/products/:id/reviews"),
        Route(method="PUT", template="/rest/products/:id/reviews"),
        Route(method="PATCH", template="/rest/products/reviews"),
        Route(method="POST", template="/rest/products/reviews"),
    ]
}

JUICE_SHOP_SCORE_BOARD = {
    "path": "/score-board",
    "routes": [
        Route(method="GET", template="/api/Challenges/"),
        Route(method="GET", template="/snippets"),
        Route(method="GET", template="/rest/admin/application-configuration"),
        Route(method="GET", template="/rest/repeat-notification?challenge="),
    ]
}

JUICE_SHOP_TRACK_RESULT = {
    "path": "/track-result",
    "routes": [
        Route(method="GET", template="/rest/track-order/:id"),
    ]
}

JUICE_SHOP_TRACK_RESULT_NEW = {
    "path": "/track-result/new",
    "routes": [
        Route(method="GET", template="/rest/track-order/:id"),
    ]
}

JUICE_SHOP_2FA_ENTER = {
    "path": "/2fa/enter",
    "routes": [
        Route(method="POST", template="/rest/2fa/verify"),
    ]
}

JUICE_SHOP_PRIVACY_SECURITY = {
    "path": "/privacy-security",
    "routes": []
}

JUICE_SHOP_PRIVACY_SECURITY_PRIVACY_POLICY = {
    "path": "/privacy-security/privacy-policy",
    "routes": [
        Route(method="GET", template="/rest/admin/application-configuration"),
    ]
}

JUICE_SHOP_PRIVACY_SECURITY_CHANGE_PASSWORD = {
    "path": "/privacy-security/change-password",
    "routes": [
        Route(method="GET", template="/rest/user/change-password?current=&new=&repeat="),
    ]
}

JUICE_SHOP_PRIVACY_SECURITY_TWO_FACTOR_AUTHENTICATION = {
    "path": "/privacy-security/two-factor-authentication",
    "routes": [
        Route(method="GET", template="/rest/2fa/status"),
        Route(method="POST", template="/rest/2fa/setup"),
        Route(method="POST", template="/rest/2fa/disable"),
        Route(method="GET", template="/rest/admin/application-configuration"),
    ]
}

JUICE_SHOP_PRIVACY_SECURITY_DATA_EXPORT = {
    "path": "/privacy-security/data-export",
    "routes": [
        Route(method="GET", template="/rest/image-captcha/"),
        Route(method="POST", template="/rest/user/data-export"),
    ]
}

JUICE_SHOP_PRIVACY_SECURITY_LAST_LOGIN_IP = {
    "path": "/privacy-security/last-login-ip",
    "routes": []
}

JUICE_SHOP_JUICY_NFT = {
    "path": "/juicy-nft",
    "routes": [
        Route(method="GET", template="/rest/web3/nftUnlocked"),
        Route(method="POST", template="/rest/web3/submitKey"),
    ]
}

JUICE_SHOP_WALLET_WEB3 = {
    "path": "/wallet-web3",
    "routes": [
        Route(method="POST", template="/rest/web3/walletExploitAddress"),
    ]
}

JUICE_SHOP_WEB3_SANDBOX = {
    "path": "/web3-sandbox",
    "routes": []
}

JUICE_SHOP_BEE_HAVEN = {
    "path": "/bee-haven",
    "routes": [
        Route(method="GET", template="/rest/web3/nftMintListen"),
        Route(method="GET", template="/api/Challenges/?key=nftMintChallenge"),
        Route(method="POST", template="/rest/web3/walletNFTVerify"),
    ]
}

JUICE_SHOP_ALL = {
    r["path"] : r["routes"] for r in [
        JUICE_SHOP_HOMEPAGE, 
        JUICE_SHOP_ADMINISTRATION, 
        JUICE_SHOP_ACCOUNTING, 
        JUICE_SHOP_ABOUT, 
        JUICE_SHOP_ADDRESS_SELECT, 
        JUICE_SHOP_ADDRESS_SAVED,
        JUICE_SHOP_ADDRESS_CREATE, 
        JUICE_SHOP_ADDRESS_EDIT, 
        JUICE_SHOP_DELIVERY_METHOD, 
        JUICE_SHOP_DELUXE_MEMBERSHIP, 
        JUICE_SHOP_SAVED_PAYMENT_METHODS, 
        JUICE_SHOP_BASKET, 
        JUICE_SHOP_ORDER_COMPLETION, 
        JUICE_SHOP_CONTACT, 
        JUICE_SHOP_PHOTO_WALL, 
        JUICE_SHOP_COMPLAIN, 
        JUICE_SHOP_ORDER_SUMMARY, 
        JUICE_SHOP_ORDER_HISTORY, 
        JUICE_SHOP_PAYMENT, 
        JUICE_SHOP_WALLET, 
        JUICE_SHOP_LOGIN, 
        JUICE_SHOP_FORGOT_PASSWORD, 
        JUICE_SHOP_RECYCLE, 
        JUICE_SHOP_REGISTER, 
        JUICE_SHOP_SEARCH, 
        JUICE_SHOP_SCORE_BOARD, 
        JUICE_SHOP_TRACK_RESULT, 
        JUICE_SHOP_TRACK_RESULT_NEW, 
        JUICE_SHOP_2FA_ENTER, 
        JUICE_SHOP_PRIVACY_SECURITY, 
        JUICE_SHOP_PRIVACY_SECURITY_PRIVACY_POLICY, 
        JUICE_SHOP_PRIVACY_SECURITY_CHANGE_PASSWORD, 
        JUICE_SHOP_PRIVACY_SECURITY_TWO_FACTOR_AUTHENTICATION, 
        JUICE_SHOP_PRIVACY_SECURITY_DATA_EXPORT, 
        JUICE_SHOP_PRIVACY_SECURITY_LAST_LOGIN_IP, 
        JUICE_SHOP_JUICY_NFT, 
        JUICE_SHOP_WALLET_WEB3, 
        JUICE_SHOP_WEB3_SANDBOX, 
        JUICE_SHOP_BEE_HAVEN
    ]
}