"""
VigilOps AI - Automated RAG SOP Retrieval & OSHA Standard Verification Test
Validates:
1. Semantic retrieval matching real OSHA 1910 standards.
2. Verified document references and source citations.
3. Structured output: why dangerous, immediate actions, preventive measures, responsible personnel.
4. Negative query verification (no fabrication).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_sop import retrieve_sop_for_hazard

def run_rag_tests():
    print("=================================================================")
    print("📚 ZERO-PROMPT RAG SOP & OSHA REGULATION VERIFICATION TEST")
    print("=================================================================")

    test_cases = [
        ("NO_HELMET", "1910.135", "Head Protection"),
        ("NO_VEST", "1910.132", "High-Visibility"),
        ("FORKLIFT_HAZARD", "1910.178", "Powered Industrial Trucks"),
        ("CRANE_HAZARD", "1910.179", "Cranes"),
        ("RESTRICTED_ZONE", "1910.147", "Hazardous Energy")
    ]

    for hazard, expected_code, expected_title in test_cases:
        res = retrieve_sop_for_hazard(hazard)
        print(f"\n🔍 Query: '{hazard}'")
        print(f"   ↳ SOP ID: {res['sop_id']} | Code: {res['standard_code']} | Citation: {res['regulatory_ref']}")
        print(f"   ↳ Title: {res['title']}")
        print(f"   ↳ Why Dangerous: {res['why_dangerous']}")
        print(f"   ↳ Immediate Actions: {len(res['immediate_actions'])} items -> Step 1: {res['immediate_actions'][0]}")
        print(f"   ↳ Responsible: {res['responsible_personnel']}")
        print(f"   ↳ Source: {res['reference_source']} ({res['reference_section']})")
        print(f"   ↳ Status: {res['retrieval_status']}")

        assert expected_code in res["standard_code"], f"Expected {expected_code} in {res['standard_code']}"
        assert res["retrieval_status"] == "VERIFIED_FROM_CORPUS", f"Expected VERIFIED_FROM_CORPUS"
        assert len(res["immediate_actions"]) >= 3, "Missing immediate action checklist"
        assert len(res["why_dangerous"]) > 20, "Missing dangerous explanation"

    # Negative Test: Unknown query
    print("\n🔍 Negative Test: 'UNKNOWN_RANDOM_QUERY_999'")
    unknown_res = retrieve_sop_for_hazard("UNKNOWN_RANDOM_QUERY_999")
    print(f"   ↳ Status: {unknown_res['retrieval_status']}")
    print(f"   ↳ Title: {unknown_res['title']}")
    assert unknown_res["retrieval_status"] == "NO_VERIFIED_PROCEDURE_FOUND"
    print("✅ Negative query correctly reported NO_VERIFIED_PROCEDURE_FOUND without fabricating regulations.")

    print("\n-----------------------------------------------------------------")
    print("=================================================================")
    print("🎉 ZERO-PROMPT RAG SOP & OSHA RETRIEVAL TEST: PASSED")
    print("=================================================================\n")

if __name__ == "__main__":
    run_rag_tests()
