"""
Example API responses showing the difference with and without Gemini AI
"""

# WITHOUT GEMINI API KEY:
example_without_gemini = {
    "input": {
        "tds": 975.0,
        "turbidity": 8.13,
        "ph": 4.0
    },
    "prediction": {
        "quality_class": 0,
        "quality_label": "Poor",
        "confidence": 0.92
    },
    "probabilities": {
        "Poor": 0.92,
        "Acceptable": 0.079,
        "Good": 0.0006,
        "Excellent": 0.0001
    },
    "recommendation": "Poor water quality. Treatment required before consumption.",
    "warnings": [
        "pH 4.0 is outside typical drinking water range (6.0-9.0)"
    ],
    "summary": "AI summary not available. Please check Gemini API configuration.",
    "timestamp": "2025-08-31T17:56:02.061847"
}

# WITH GEMINI API KEY CONFIGURED:
example_with_gemini = {
    "input": {
        "tds": 975.0,
        "turbidity": 8.13,
        "ph": 4.0
    },
    "prediction": {
        "quality_class": 0,
        "quality_label": "Poor",
        "confidence": 0.92
    },
    "probabilities": {
        "Poor": 0.92,
        "Acceptable": 0.079,
        "Good": 0.0006,
        "Excellent": 0.0001
    },
    "recommendation": "Poor water quality. Treatment required before consumption.",
    "warnings": [
        "pH 4.0 is outside typical drinking water range (6.0-9.0)"
    ],
    "summary": "This water sample shows serious quality concerns that make it unsafe for drinking without treatment. The extremely acidic pH of 4.0 is far below safe drinking levels and could be harmful to your health, while the high dissolved solids content of 975 mg/L exceeds recommended limits. I strongly recommend using a proper water treatment system or finding an alternative safe water source before consuming this water.",
    "timestamp": "2025-08-31T17:56:02.061847"
}

print("🔄 COMPARISON: API Response Without vs With Gemini AI")
print("=" * 70)

print("\n❌ WITHOUT GEMINI:")
print(f"Summary: {example_without_gemini['summary']}")

print("\n✅ WITH GEMINI:")  
print(f"Summary: {example_with_gemini['summary']}")

print("\n🎯 KEY BENEFITS OF GEMINI INTEGRATION:")
print("• Human-readable explanations in plain language")
print("• Contextual health and safety advice")
print("• Consumer-friendly recommendations")
print("• Professional water quality assessment tone")
print("• Actionable guidance for water treatment")
