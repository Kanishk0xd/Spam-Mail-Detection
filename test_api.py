import requests
import json

# Configuration
API_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")

def test_single_prediction():
    """Test single email prediction"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    test_cases = [
        {
            "text": "Congratulations! You have won a $1000 prize. Click here now!",
            "expected": "SPAM"
        },
        {
            "text": "Hi, can we schedule a meeting for tomorrow at 3pm?",
            "expected": "HAM"
        },
        {
            "text": "URGENT: Your account will be suspended! Verify now!",
            "expected": "SPAM"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {test_case['text'][:60]}...")
        
        response = requests.post(
            f"{API_URL}/predict_spam",
            json={"text": test_case["text"]}
        )
        
        result = response.json()
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Expected: {test_case['expected']}")
        
        if result['label'] == test_case['expected']:
            print("✅ Prediction correct!")
        else:
            print("⚠️  Prediction differs from expected")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    emails = [
        "Win a free iPhone now! Limited offer!",
        "The quarterly report is ready for review.",
        "Make money fast! Work from home!",
        "Meeting rescheduled to 2pm tomorrow.",
        "URGENT! Your package is waiting!"
    ]
    
    response = requests.post(
        f"{API_URL}/predict_spam_batch",
        json={"emails": emails}
    )
    
    result = response.json()
    print(f"\nTotal Emails: {result['total_emails']}")
    print(f"Spam Detected: {result['spam_count']}")
    print(f"Ham Detected: {result['ham_count']}")
    
    print("\nDetailed Results:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"\n{i}. {pred['text'][:50]}...")
        print(f"   Label: {pred['label']} (Confidence: {pred['confidence']:.2%})")
    
    print("\n Batch prediction completed!")

def test_api_docs():
    """Test API documentation"""
    print("\n" + "="*60)
    print("Testing API Documentation")
    print("="*60)
    
    response = requests.get(f"{API_URL}/docs")
    print(f"Status Code: {response.status_code}")
    print(f"API Docs available at: {API_URL}/docs")
    print("API documentation accessible!")

def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    # Empty text
    print("\n1. Testing empty text:")
    response = requests.post(
        f"{API_URL}/predict_spam",
        json={"text": ""}
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print("Correctly rejected empty text")
    
    # Very long text
    print("\n2. Testing very long text:")
    long_text = "spam " * 1000
    response = requests.post(
        f"{API_URL}/predict_spam",
        json={"text": long_text}
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Successfully handled long text")
    
    # Special characters
    print("\n3. Testing special characters:")
    special_text = "¡Hola! 你好 مرحبا @#$%^&*()"
    response = requests.post(
        f"{API_URL}/predict_spam",
        json={"text": special_text}
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Successfully handled special characters")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SPAM DETECTION API - TEST SUITE")
    print("="*60)
    print(f"Testing API at: {API_URL}")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_api_docs()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n ERROR: Cannot connect to API")
        print(f"Make sure the server is running at {API_URL}")
        print("Run: uvicorn app.main:app --host 0.0.0.0 --port 8080")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")

if __name__ == "__main__":
    run_all_tests()