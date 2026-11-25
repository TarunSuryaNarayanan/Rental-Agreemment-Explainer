"""Four-part prompt anatomy: INPUT CONTEXT, SYSTEM INSTRUCTIONS, OUTPUT CONSTRAINTS, FEW-SHOT.

Strategies: zero_shot | persona | template | few_shot | cot
Each strategy toggles which parts are included, so strategies can be compared fairly.
"""
import json
from dataclasses import dataclass

CLAUSE_TYPES = [
    "rent", "deposit", "lock_in", "maintenance", "termination", "notice_period",
    "rent_increase", "rent_escalation", "penalty", "landlord_entry", "subletting", "utilities",
    "use_of_premises", "renewal", "dispute_resolution", "police_verification", "registration", "other",
]
LABELS = ["Standard", "Unusual", "One-sided"]
STRATEGIES = ["zero_shot", "persona", "template", "few_shot", "cot", "self_consistency"]

# ---- 2. SYSTEM INSTRUCTIONS (several paraphrases for the sensitivity study) ----
INSTRUCTION_VARIANTS = [
    "You explain one clause of a residential rental agreement to a first-time tenant. "
    "Use only the clause text provided. Never invent terms. Do not give legal verdicts.",
    "Your job is to help a first-time renter understand a single clause from their lease. "
    "Rely strictly on the supplied clause; do not add terms that are absent, and avoid legal conclusions.",
    "Act as a careful assistant that reads one lease clause and tells a new tenant what it means. "
    "Stay faithful to the given wording, make nothing up, and refrain from declaring anything legal or illegal.",
]

PERSONA = (
    "You are a friendly tenant-rights volunteer with 15 years of experience helping students "
    "and young professionals read rental agreements. You are not a lawyer."
)

LABEL_DEFINITIONS = (
    "Label definitions & Indian tenancy standards:\n"
    "- Standard: common, balanced clause typically seen in Indian residential agreements.\n"
    "    * Subletting prohibitions or restrictions without landlord consent are STANDARD.\n"
    "    * Specifying court jurisdiction, dispute resolution, or arbitration is STANDARD.\n"
    "    * Renewal terms subject to mutual written consent/renegotiation are STANDARD.\n"
    "    * Lock-in periods (e.g. 1 to 6 months) and mutual notice periods (e.g. 30 to 60 days) are STANDARD.\n"
    "    * Standard tenant covenants (routine consumable maintenance, paying utility bills, strictly residential use) are STANDARD.\n"
    "    * Police verification and stamp duty/registration provisions are STANDARD.\n"
    "    * Do not classify a clause as Incomplete or Unusual simply because additional optional details could improve it.\n"
    "    * Incomplete drafts: If a clause contains template placeholders, blanks, or unpopulated values (e.g., 'Rs. [Amount]', '_____', or unspecified duration), "
    "it is an INCOMPLETE template field, NOT an unusual clause. Label it Standard (or whatever fits the underlying term) and set \"is_incomplete\": true.\n"
    "- Unusual: uncommon, stricter-than-normal, or burdensome provisions in Indian leases, even if mutually applicable.\n"
    "    * Examples: requiring 6+ months advance rent upfront, forcing tenant to pay exterior/structural repairs, "
    "unreasonable guest or visitor curfews, mandatory painting fees deducted irrespective of handover condition.\n"
    "- One-sided: gives the landlord disproportionate rights/remedies with little or no reciprocal obligation, "
    "or imposes heavy unilateral penalties on the tenant only.\n"
    "    * Examples: landlord entering the premises at any time without prior notice, total deposit forfeiture for minor defaults, "
    "landlord determining repair deductions at 'sole discretion' without bills or proof, asymmetrical termination notice periods."
)

# ---- 3. OUTPUT CONSTRAINTS ----
OUTPUT_CONSTRAINTS = (
    "Respond with ONLY a JSON object, no prose, with these keys:\n"
    f'  "clause_type": one of {CLAUSE_TYPES}\n'
    '  "explanation": 1-3 plain-English sentences, reading level ~Grade 8\n'
    f'  "label": one of {LABELS}\n'
    '  "is_incomplete": boolean (true ONLY if the clause contains unfilled placeholders, blanks [___], or missing terms to be filled; otherwise false)\n'
    '  "label_reason": one line justifying the label (if incomplete, note what detail is missing)\n'
    '  "quote": an EXACT, word-for-word substring of the clause that justifies your label '
    "(copy it, do not paraphrase)\n"
)
COT_EXTRA = (
    '  "reasoning": brief step-by-step thinking (what each party must do, who bears risk) '
    "BEFORE you decide the label\n"
)
TEMPLATE_EXTRA = (
    "Follow this template mentally: (1) Who must do what? (2) By when / how much? "
    "(3) What happens if it is breached? Then write the explanation.\n"
)

# ---- 4. FEW-SHOT EXAMPLES ----
FEW_SHOT = [
    {
        "clause": "The Security Deposit of Rs. 1,00,000 shall be refunded within 30 days of vacating, "
                  "after deducting only actual damages beyond normal wear and tear.",
        "out": {
            "clause_type": "deposit",
            "explanation": "You pay Rs. 1,00,000 upfront. You get it back within 30 days after moving out, "
                           "minus the cost of real damage (not normal wear).",
            "label": "Standard",
            "is_incomplete": False,
            "label_reason": "Clear refund timeline and deductions limited to actual damage.",
            "quote": "refunded within 30 days of vacating",
        },
    },
    {
        "clause": "The Lessee shall not sublet, assign, or part with the possession of the Premises or any portion thereof without the prior written consent of the Lessor.",
        "out": {
            "clause_type": "subletting",
            "explanation": "You cannot rent out or transfer the flat to anyone else without getting the landlord's written permission first.",
            "label": "Standard",
            "is_incomplete": False,
            "label_reason": "Prohibiting unauthorized subletting is standard protective boilerplate in residential leases.",
            "quote": "not sublet, assign, or part with the possession",
        },
    },
    {
        "clause": "The Lessee shall pay a monthly rent of Rs. [Amount] on or before the [Day] of each English calendar month.",
        "out": {
            "clause_type": "rent",
            "explanation": "You must pay your monthly rent by the agreed date each month. The exact amount and due date still need to be filled into this draft.",
            "label": "Standard",
            "is_incomplete": True,
            "label_reason": "Standard monthly rent provision, but the amount and due date are blank template placeholders.",
            "quote": "monthly rent of Rs. [Amount]",
        },
    },
    {
        "clause": "If the Lessee vacates before 24 months, the entire Security Deposit shall stand forfeited "
                  "and the Lessor may recover further losses at his sole discretion.",
        "out": {
            "clause_type": "penalty",
            "explanation": "If you leave before 24 months you lose your whole deposit, and the landlord can "
                           "also claim more money from you however he decides.",
            "label": "One-sided",
            "is_incomplete": False,
            "label_reason": "Full forfeiture plus open-ended extra liability on tenant only.",
            "quote": "entire Security Deposit shall stand forfeited",
        },
    },
]


@dataclass
class Prompt:
    system: str
    user: str


def build_prompt(clause: str, strategy: str = "few_shot", variant: int = 0, doc_context: str = "") -> Prompt:
    instr = INSTRUCTION_VARIANTS[variant % len(INSTRUCTION_VARIANTS)]
    system_parts = []
    if strategy == "persona":
        system_parts.append(PERSONA)
    system_parts.append(instr)
    system_parts.append(LABEL_DEFINITIONS)
    if strategy == "template":
        system_parts.append(TEMPLATE_EXTRA)
    system_parts.append(OUTPUT_CONSTRAINTS + (COT_EXTRA if strategy in ("cot", "self_consistency") else ""))
    system = "\n\n".join(system_parts)

    user_parts = []
    if strategy in ("few_shot", "self_consistency"):
        for ex in FEW_SHOT:
            user_parts.append(f"<clause>{ex['clause']}</clause>\n{json.dumps(ex['out'])}")
        user_parts.append("Now the real clause:")
    # 1. INPUT CONTEXT
    if doc_context:
        user_parts.append(f"<document_summary>{doc_context}</document_summary>")
    user_parts.append(f"<clause>{clause}</clause>")
    return Prompt(system=system, user="\n\n".join(user_parts))


def build_retry_prompt(base: Prompt, bad_quote: str) -> Prompt:
    note = (
        f'\n\nYour previous "quote" ("{bad_quote}") does not appear verbatim in the clause. '
        "Copy an exact substring from the clause and answer again."
    )
    return Prompt(system=base.system, user=base.user + note)
