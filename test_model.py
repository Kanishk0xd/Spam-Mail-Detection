import joblib

# Load model
pipeline = joblib.load('models/spam_pipeline.joblib')

# Test emails
test_emails = [
    "Win free prizes now! Click here!",
    "Can we schedule a meeting tomorrow at 3pm?",
    "URGENT: Your account will be suspended! Verify now!",
    "Thanks for the project update. Looks good.",
    "Make money fast! Work from home!",
    "See you at the conference next week."
]

print("=" * 60)
print("TESTING SPAM DETECTION MODEL")
print("=" * 60)

for i, text in enumerate(test_emails, 1):
    result = pipeline.predict([text])[0]
    prob = pipeline.predict_proba([text])[0]
    label = 'SPAM' if result == 1 else 'HAM'
    confidence = prob[result]
    
    # Emoji indicator
    icon = 'Spam' if label == 'SPAM' else 'Ham'
    
    print(f"\n{i}. {icon} {text[:50]}...")
    print(f"   Prediction: {label}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Spam prob: {prob[1]:.2%} | Ham prob: {prob[0]:.2%}")

print("\n" + "=" * 60)