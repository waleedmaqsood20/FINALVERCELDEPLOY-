import requests
import json

# Your API URL
API_URL = "https://modelv2-k7ncumuok-waleedmaqsood20s-projects.vercel.app"

def test_health():
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json() if response.status_code == 200 else response.text[:100]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_generate_quick():
    print("\nTesting quick text generation...")
    data = {
        "prompt": "Write a short poem about coding",
        "max_length": 50,  # Much smaller length to avoid timeouts
        "temperature": 0.7
    }
    try:
        response = requests.post(f"{API_URL}/generate-quick", json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nGenerated Text:")
            print(f"{result.get('generated_text')}")
            return True
        else:
            print(f"Response: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_generate_regular():
    print("\nTesting regular text generation (may timeout)...")
    data = {
        "prompt": "Write a short poem about coding",
        "max_length": 50,  # Use smaller length to avoid timeouts
        "temperature": 0.7
    }
    try:
        response = requests.post(f"{API_URL}/generate", json=data, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nGenerated Text:")
            print(f"{result.get('generated_text')}")
            return True
        else:
            print(f"Response: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Testing API at: {API_URL}")
    
    health_ok = test_health()
    quick_ok = test_generate_quick()
    regular_ok = test_generate_regular()
    
    print("\nTest Results:")
    print(f"Health Check: {'✅ PASSED' if health_ok else '❌ FAILED'}")
    print(f"Quick Generation: {'✅ PASSED' if quick_ok else '❌ FAILED'}")
    print(f"Regular Generation: {'✅ PASSED' if regular_ok else '❌ FAILED'}")
    
    if health_ok and (quick_ok or regular_ok):
        print("\n🎉 Tests passed! Your API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the details above for more information.") 