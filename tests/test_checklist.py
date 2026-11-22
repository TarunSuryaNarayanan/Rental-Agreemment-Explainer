import pytest
from rae.checklist import (
    CATEGORY_MAPPING,
    CANONICAL_KEY_TERMS,
    CHECKLIST_SPECS,
    check_missing,
    validate_evidence,
    extract_section_ref,
)
from rae.splitter import guess_type_keywords
from rae.verifier import quote_in_text

EXPECTED_CANONICAL_DESCRIPTIONS = {
    "rent": "Rent amount, payment frequency and due date.",
    "deposit": "Security deposit amount, deductions and refund timeline.",
    "lock_in": "Minimum mandatory tenancy period and early-exit restrictions.",
    "notice_period": "Required advance notice before termination.",
    "termination": "Conditions and procedures for ending the agreement.",
    "rent_increase": "Rent escalation percentage, timing and conditions.",
    "maintenance": "Tenant and landlord repair responsibilities.",
    "landlord_entry": "Landlord access, inspection and advance notice.",
    "subletting": "Restrictions on subletting or transferring possession.",
    "utilities": "Responsibility for electricity, water, gas and other bills.",
    "dispute_resolution": "Dispute settlement and court jurisdiction.",
    "police_verification": "Tenant police verification requirements.",
    "registration": "Stamp duty and registration requirements.",
}

EXPECTED_CATEGORY_MAPPING = {
    "MONTHLY RENT": "rent",
    "SECURITY DEPOSIT": "deposit",
    "LOCK-IN PERIOD": "lock_in",
    "NOTICE PERIOD": "notice_period",
    "RENT INCREASE": "rent_increase",
    "MAINTENANCE AND REPAIRS": "maintenance",
    "LANDLORD ENTRY AND INSPECTION": "landlord_entry",
    "SUBLETTING": "subletting",
    "UTILITIES AND OTHER CHARGES": "utilities",
    "TERMINATION CONDITIONS": "termination",
    "POLICE VERIFICATION": "police_verification",
    "REGISTRATION AND STAMP DUTY": "registration",
    "DISPUTE RESOLUTION": "dispute_resolution",
}


def test_canonical_descriptions_match_specification():
    """Validate that every key's description matches its key identifier exactly."""
    assert len(CANONICAL_KEY_TERMS) == 13
    for key, expected_desc in EXPECTED_CANONICAL_DESCRIPTIONS.items():
        assert key in CANONICAL_KEY_TERMS, f"Missing key {key}"
        assert CANONICAL_KEY_TERMS[key] == expected_desc, f"Mismatch for key {key}"


def test_exact_category_mappings():
    """Validate exact 13 category-to-key mappings specified by the user."""
    assert len(CATEGORY_MAPPING) == 13
    for cat_name, expected_key in EXPECTED_CATEGORY_MAPPING.items():
        assert cat_name in CATEGORY_MAPPING, f"Missing category {cat_name}"
        assert CATEGORY_MAPPING[cat_name] == expected_key, f"Mismatch for {cat_name}"


def test_clause_category_identifiers_match_actual_sections():
    """Ensure clause category identifiers match their actual sections."""
    assert guess_type_keywords("4. NOTICE PERIOD: Either party may terminate after lock-in with 30 days notice.") == "notice_period"
    assert guess_type_keywords("9. UTILITIES AND OTHER CHARGES: Tenant shall pay electricity and water bills.") == "utilities"
    assert guess_type_keywords("7. LANDLORD ENTRY AND INSPECTION: Landlord may enter upon 24 hours notice.") == "landlord_entry"
    assert guess_type_keywords("12. REGISTRATION AND STAMP DUTY: Stamp duty and registration charges shall be shared.") == "registration"
    assert guess_type_keywords("10. TERMINATION CONDITIONS: In case of default or material breach, lease may be terminated.") == "termination"


def test_each_key_uses_own_unique_description():
    """Validate that no two keys share the same description."""
    descriptions = list(CANONICAL_KEY_TERMS.values())
    assert len(descriptions) == len(set(descriptions)), "Duplicate descriptions found across keys"


def test_clearly_specified_three_month_lock_in_is_present():
    """A clearly specified three-month lock-in period must be classified as Present."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. LOCK-IN PERIOD: Both parties agree to a mandatory lock-in period of 3 months "
        "from the commencement date. Neither party shall terminate during this time."
    )
    result = check_missing(doc)
    item = result["items"]["lock_in"]
    assert item["status"] == "Present"
    assert "lock_in" in result["present"]
    assert quote_in_text(item["quote"], doc)


def test_termination_clause_is_present():
    """A termination clause specifying valid termination grounds and notice requirements should be Present."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. TERMINATION CONDITIONS: Either party may terminate this agreement upon 30 days written notice "
        "or immediately in case of material breach of covenants."
    )
    result = check_missing(doc)
    item = result["items"]["termination"]
    assert item["status"] == "Present"
    assert "termination" in result["present"]
    assert item["quote"] != ""
    assert quote_in_text(item["quote"], doc)


def test_never_use_property_details_as_evidence_for_police_or_registration():
    """Never use PROPERTY DETAILS as evidence for police verification or registration."""
    doc = (
        "RENTAL AGREEMENT\n"
        "This Agreement is made for the residential premises at 12 MG Road, Bengaluru, "
        "measuring about 1200 sq.ft, bounded on North by road, situated within the jurisdiction "
        "of Cubbon Park Police Station limits, registered as Document No. 5678/2024 at Sub-Registrar Office.\n"
        "1. MONTHLY RENT: The Lessee shall pay Rs. 20,000 per month.\n"
        "2. SECURITY DEPOSIT: Deposit of Rs. 1,00,000 paid to Lessor."
    )
    result = check_missing(doc)
    pv = result["items"]["police_verification"]
    reg = result["items"]["registration"]

    assert pv["status"] == "Missing", "Property details must not trigger police verification"
    assert pv["quote"] == ""
    assert reg["status"] == "Missing", "Property details/sale deed recitals must not trigger registration"
    assert reg["quote"] == ""


def test_police_verification_and_registration_valid_evidence():
    """Police verification and registration match their genuine clauses separately."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. POLICE VERIFICATION: Section 14 - The Lessee shall complete tenant police verification at the local station within 15 days.\n"
        "2. REGISTRATION AND STAMP DUTY: Section 17 - Stamp duty and registration charges shall be borne equally by both parties."
    )
    res = check_missing(doc)
    pv = res["items"]["police_verification"]
    reg = res["items"]["registration"]

    assert pv["status"] == "Present"
    assert "Section 14" in pv["section_ref"]
    assert "police verification" in pv["quote"].lower()
    assert quote_in_text(pv["quote"], doc)

    assert reg["status"] == "Present"
    assert "Section 17" in reg["section_ref"]
    assert "stamp duty and registration" in reg["quote"].lower()
    assert quote_in_text(reg["quote"], doc)


def test_rent_increase_evidence_precision():
    """Rent increases must reference both escalation condition and applicable renewal terms."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. RENT INCREASE: Section 5 - The monthly rent shall be increased by 5% upon renewal at the end of every 11 months."
    )
    res = check_missing(doc)
    ri = res["items"]["rent_increase"]
    assert ri["status"] == "Present"
    assert "5%" in ri["quote"]
    assert "renewal" in ri["quote"].lower()
    assert quote_in_text(ri["quote"], doc)


def test_police_verification_distinguishes_id_collection_from_completed_verification():
    """Police verification must distinguish identity-document collection from completed police verification."""
    doc_id_only = (
        "RENTAL AGREEMENT\n"
        "1. POLICE VERIFICATION: The Lessee shall submit identity documents and Aadhaar card copies to the Lessor."
    )
    res_id = check_missing(doc_id_only)
    assert res_id["items"]["police_verification"]["verification_type"] == "Identity Document Collection Only"

    doc_full = (
        "RENTAL AGREEMENT\n"
        "1. POLICE VERIFICATION: The Lessee shall complete tenant police verification formalities at the local police station."
    )
    res_full = check_missing(doc_full)
    assert res_full["items"]["police_verification"]["verification_type"] == "Completed Police Verification at Local Station"


def test_registration_distinguishes_promise_to_register_from_proof_of_actual_registration():
    """Registration must distinguish a contractual promise to register from proof of actual registration."""
    doc_promise = (
        "RENTAL AGREEMENT\n"
        "1. REGISTRATION AND STAMP DUTY: Stamp duty and registration charges shall be shared equally between the parties."
    )
    res_promise = check_missing(doc_promise)
    assert res_promise["items"]["registration"]["registration_type"] == "Contractual Promise to Register"

    doc_proof = (
        "RENTAL AGREEMENT\n"
        "1. REGISTRATION AND STAMP DUTY: This agreement has been registered as Document No. 8910/2026 in Book I."
    )
    res_proof = check_missing(doc_proof)
    assert res_proof["items"]["registration"]["registration_type"] == "Proof of Actual Registration"


def test_concept_separation():
    """Keep clause presence, clause completeness, legal compliance, and risk level separate."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. MONTHLY RENT: Rs. 25,000 per month."
    )
    res = check_missing(doc)
    item = res["items"]["rent"]
    assert item["presence"] == "Present"
    assert item["completeness"] == "Complete"
    assert item["legal_compliance"] == "Compliant"
    assert item["risk_level"] == "Standard"


def test_incomplete_clause_preserves_supporting_quotation():
    """Incomplete clauses must still have relevant supporting quotations."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. MONTHLY RENT: The Lessee shall pay a monthly rent of Rs. [Amount] on or before the 5th.\n"
        "2. LOCK-IN PERIOD: There shall be a mandatory lock-in period of _____ months."
    )
    res = check_missing(doc)
    rent = res["items"]["rent"]
    lock_in = res["items"]["lock_in"]

    assert rent["status"] == "Incomplete"
    assert rent["quote"] != ""
    assert "[Amount]" in rent["quote"]
    assert quote_in_text(rent["quote"], doc)

    assert lock_in["status"] == "Incomplete"
    assert lock_in["quote"] != ""
    assert "_____" in lock_in["quote"]
    assert quote_in_text(lock_in["quote"], doc)


def test_do_not_classify_as_incomplete_for_optional_details():
    """Do not classify a clause as Incomplete simply because additional optional details could improve it."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. MAINTENANCE AND REPAIRS: The Lessee shall maintain the premises in good condition and bear routine minor repairs."
    )
    result = check_missing(doc)
    assert result["items"]["maintenance"]["status"] == "Present"
    assert "maintenance" in result["present"]


def test_checklist_status_and_key_term_summary_agree():
    """Ensure checklist status and key-term summary status agree across all 13 keys."""
    doc = (
        "RENTAL AGREEMENT\n"
        "1. MONTHLY RENT: Rs. 25,000 per month.\n"
        "2. SECURITY DEPOSIT: Rs. 1,00,000 refundable.\n"
        "3. LOCK-IN PERIOD: Mandatory 3 months lock-in period.\n"
        "4. NOTICE PERIOD: 30 days written notice to terminate.\n"
        "5. RENT INCREASE: 5% annual increase.\n"
        "6. MAINTENANCE AND REPAIRS: Tenant handles minor repairs, Landlord major.\n"
        "7. LANDLORD ENTRY AND INSPECTION: Entry permitted after 24 hours notice.\n"
        "8. SUBLETTING: Subletting strictly prohibited without consent.\n"
        "9. UTILITIES AND OTHER CHARGES: Tenant pays electricity and water.\n"
        "10. TERMINATION CONDITIONS: Agreement terminable on material breach.\n"
        "11. POLICE VERIFICATION: Tenant verification at police station.\n"
        "12. REGISTRATION AND STAMP DUTY: Registration charges shared equally.\n"
        "13. DISPUTE RESOLUTION: Exclusive jurisdiction of civil courts at Bengaluru."
    )
    res = check_missing(doc)
    for kt in res["key_terms"]:
        k = kt["key"]
        st = kt["status"]
        if st == "Present":
            assert k in res["present"], f"{k} marked Present in key_terms but missing from present list"
        elif st == "Incomplete":
            assert k in res["incomplete"], f"{k} marked Incomplete in key_terms but missing from incomplete list"
        else:
            assert k in res["missing"], f"{k} marked Missing in key_terms but missing from missing list"
