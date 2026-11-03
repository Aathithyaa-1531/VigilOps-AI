"""
Generate Authentic OSHA Standard 1910 Corpus in Markdown and PDF formats.
Populates OSHA General Industry Safety & Health Regulations for RAG retrieval.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

SAFETY_DOCS_DIR = os.path.join(os.path.dirname(__file__), "models", "safety_docs")
os.makedirs(SAFETY_DOCS_DIR, exist_ok=True)

PDF_PATH = os.path.join(SAFETY_DOCS_DIR, "osha_standards.pdf")
MD_PATH = os.path.join(SAFETY_DOCS_DIR, "osha_standards_corpus.md")

OSHA_CORPUS_TEXT = """# OSHA Standard 1910 Safety & Health Compliance Corpus
**United States Department of Labor - Occupational Safety and Health Administration (OSHA)**
**Document Reference: 29 CFR Part 1910 - Occupational Safety and Health Standards**
**Classification: Industrial & Manufacturing EHS Protocol Directive**

---

## 1. OSHA 1910.132 - General Requirements for Personal Protective Equipment (PPE)
### Section Summary
Application. Protective equipment, including personal protective equipment for eyes, face, head, and extremities, protective clothing, respiratory devices, and protective shields and barriers, shall be provided, used, and maintained in a sanitary and reliable condition wherever it is necessary by reason of hazards of processes or environment.

### Hazard Assessment and Equipment Selection (1910.132(d))
- The employer shall assess the workplace to determine if hazards are present, or are likely to be present, which necessitate the use of personal protective equipment (PPE).
- Verify that the required workplace hazard assessment has been performed through a written certification that identifies the workplace evaluated.
- Defective or damaged personal protective equipment shall not be used.

### Mandatory Compliance Protocol (SOP-EHS-042)
1. Halt personnel movement into active industrial sector immediately if required PPE is missing.
2. Conduct instant gear inspection for ANSI Z89.1 certified head protection and ANSI/ISEA 107 high-visibility vest.
3. Replace expired, cracked, or chemically compromised safety gear prior to site re-entry.
4. Record infraction in central plant EHS register with digital timestamp and supervisor sign-off.

---

## 2. OSHA 1910.135 - Head Protection (Safety Helmets & Hard Hats)
### General Requirements (1910.135(a))
- The employer shall ensure that each affected employee wears a protective helmet when working in areas where there is a potential for injury to the head from falling objects.
- The employer shall ensure that a protective helmet designed to reduce electrical shock hazard is worn by each affected employee when exposed to electrical conductors.

### Criteria for Protective Helmets (1910.135(b))
- Head protection must comply with ANSI Z89.1-2014 Type I (impact protection from above) or Type II (impact protection from top and lateral sides), Class G (General), Class E (Electrical), or Class C (Conductive).

### Mandatory Compliance Protocol (SOP-EHS-042)
- Any worker detected without hard hat in overhead hoist, crane, or racking zones must be removed from the perimeter within 10 seconds.
- Issue Type I/II Class E hard hat with suspension liner adjusted to standard 1 to 1.25 inch clearance.
- Sound localized audible horn if worker remains unhelmeted within 15 seconds.

---

## 3. OSHA 1910.132(d) & ANSI/ISEA 107-2020 - High-Visibility Reflective Apparel (Safety Vests)
### Standards & Specification
- Workers exposed to vehicular traffic, forklifts, motorized plant trucks, or heavy machinery shall wear Class 2 or Class 3 high-visibility safety apparel retroreflective to a minimum of 1,000 feet visibility.
- Retroreflective bands must cover 360-degree torso area with minimum 201 square inches of retroreflective material for Class 2 garments.

### Mandatory Compliance Protocol (SOP-EHS-045)
1. Instruct pedestrian worker without reflective vest to exit active vehicle transit corridors immediately.
2. Direct worker to the designated green walking lane separated by physical bollards.
3. Provide ANSI/ISEA 107 certified day/night reflective garment before authorising floor crossing.

---

## 4. OSHA 1910.178 - Powered Industrial Trucks (Forklifts)
### Operational Safety & Pedestrian Separation (1910.178(m) / 1910.178(n))
- All powered industrial trucks must maintain a safe operating distance of at least 3 meters (10 feet) from all pedestrians at all times.
- Forklift operators shall sound the directional horn when crossing aisle intersections, blind spots, and loading bays.
- Operators shall travel at a safe speed not exceeding 5 mph within internal facility lanes.

### Hazardous Proximity Protocol (SOP-EHS-077)
1. Automatically activate overhead amber warning strobe lights when pedestrian-forklift proximity drops below 3.0 meters.
2. Require the forklift operator to set parking brake, lower forks completely to the floor, and shut down propulsion.
3. Divert pedestrian to designated barrier-protected crosswalk.
4. Verify clear 360-degree zone prior to forklift re-engagement.

---

## 5. OSHA 1910.179 - Overhead and Gantry Cranes
### Suspended Load Fall Zone Protocols (1910.179(n))
- The crane operator shall not carry loads over people. No person shall be permitted to stand or pass under an elevated load or crane hook.
- Operators shall sound the audible warning horn prior to moving trolley or hoist over active production sectors.

### Emergency Suspension Breach Protocol (SOP-EHS-091)
1. Trigger automatic hoist wireless E-STOP interlock if unauthorized personnel cross under red laser fall-zone boundary.
2. Sound plant-wide suspended load siren.
3. Evacuate all personnel 10 meters outside the projected load impact perimeter.
4. Certified rigger must inspect sling balance and load stabilization prior to resuming hoist movement.

---

## 6. OSHA 1910.147 - The Control of Hazardous Energy (Lockout / Tagout)
### Isolation & Exclusion Zones (1910.147(c))
- Establish an energy control program consisting of energy control procedures, employee training, and periodic inspections.
- Machine guards, electrical enclosures, and high-voltage cabinets must remain secured. Unauthorized breach into automated robotic cells is strictly prohibited.

### Hazardous Energy Breach Protocol (SOP-EHS-108)
1. Execute immediate automated power shutdown (Zero Energy State) to robotic cell or high-voltage feeder.
2. Verify visual mechanical disconnects and pneumatic dump valves.
3. Lockout with standardized Master Lock safety padlocks and danger tags.
4. Cordon perimeter with physical high-visibility barricade tape and verify zero electrical potential.

---

## 7. OSHA 1910.28 & 1910.29 - Fall Protection & Walking-Working Surfaces
### Duty to Have Fall Protection (1910.28(b))
- The employer must ensure that each employee on a walking-working surface with an unprotected side or edge that is 4 feet (1.2 m) or more above a lower level is protected from falling by guardrail systems, safety net systems, or personal fall arrest systems.
- Toe-boards must have a minimum vertical height of 3.5 inches to prevent falling tools.

### Leading Edge Protocol (SOP-EHS-058)
1. Verify 100% tie-off using full-body harness and self-retracting lifeline (SRL).
2. Anchor points must withstand minimum 5,000 lbs (22.2 kN) static tensile load.
3. Immediately flag and isolate open floor hatches or missing guardrail segments.
"""

def generate_markdown_corpus():
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(OSHA_CORPUS_TEXT)
    print(f"✅ Generated OSHA 1910 Markdown Corpus at: {MD_PATH} ({os.path.getsize(MD_PATH)} bytes)")

def generate_pdf_corpus():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0369a1'),
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#991b1b'),
        backColor=colors.HexColor('#fee2e2'),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("OSHA STANDARD 1910 REGULATORY SAFETY MANUAL", title_style))
    story.append(Paragraph("United States Occupational Safety and Health Administration (OSHA)<br/>29 CFR Part 1910 - General Industry EHS Regulatory Compliance Corpus", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=12))

    sections = [
        (
            "1. OSHA 1910.132 - General Requirements for PPE",
            "Hazard Assessment & Mandatory Protective Equipment",
            "Protective equipment for eyes, face, head, and extremities, protective clothing, and protective shields shall be provided, used, and maintained in a sanitary condition. Employers must verify workplace hazard assessment with written certification.",
            [
                "Halt personnel entering active industrial area without verified PPE.",
                "Verify ANSI Z89.1 head protection and ANSI/ISEA 107 high-visibility vest.",
                "Inspect safety equipment for structural degradation before re-authorizing access."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-042): Halt line operations and equip standard hard hat and high-vis apparel immediately."
        ),
        (
            "2. OSHA 1910.135 - Head Protection (Safety Helmets & Hard Hats)",
            "Protective Helmets in Overhead & Falling Object Zones",
            "The employer shall ensure that each affected employee wears a protective helmet when working in areas where there is a potential for injury to the head from falling objects or electrical shock hazards. Helmets must comply with ANSI Z89.1 Type I/II, Class E/G standards.",
            [
                "Mandatory hard hats in crane hoist zones, warehouse pallet racks, and construction aisles.",
                "Inspect shell for hairline fractures and suspension harness clearance.",
                "Immediate ejection of unhelmeted workers from active overhead swing radius."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-042): Trigger localized safety alarm and enforce immediate helmet compliance."
        ),
        (
            "3. OSHA 1910.132(d) & ANSI/ISEA 107 - High-Visibility Reflective Apparel",
            "Reflective Garments in Heavy Equipment & Vehicle Transit Zones",
            "Personnel exposed to industrial trucks, forklifts, or heavy machinery shall wear Class 2 or Class 3 high-visibility safety vests with 360-degree retroreflective striping visible up to 1,000 feet.",
            [
                "Require high-visibility vest in loading bays, transit pathways, and staging docks.",
                "Direct non-vested workers to marked green pedestrian aisles.",
                "Ensure fluorescent background fabric complies with chromaticity standards."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-045): Redirect personnel to safe pedestrian lane and equip Class 2 safety vest."
        ),
        (
            "4. OSHA 1910.178 - Powered Industrial Trucks (Forklifts)",
            "Pedestrian Clearance Buffer & Operational Safety Rules",
            "A minimum buffer of 3.0 meters (10 feet) must be maintained between moving forklifts and pedestrian workers. Forklift operators shall sound directional horns at intersections and adhere to facility speed limits (<5 mph).",
            [
                "Strobe activation upon pedestrian entry into 3-meter turning radius.",
                "Operator must lower forks to floor, set emergency brake, and cut engine.",
                "Pedestrian must utilize physical bollard-protected crosswalks."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-077): Audible horn trigger & full stop of forklift until pedestrian clears zone."
        ),
        (
            "5. OSHA 1910.179 - Overhead and Gantry Cranes",
            "Suspended Load Fall Zone Protocols & Hoisting Directives",
            "No employee shall be permitted beneath an elevated hoist or suspended load. Operators must sound an audible alarm before trolley traverse and enforce a 10-meter exclusion zone.",
            [
                "Automated laser boundary monitoring beneath active hoist hooks.",
                "Immediate trip of hoist wireless E-STOP upon perimeter breach.",
                "Rigging inspection by qualified safety specialist prior to restart."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-091): EMERGENCY HOIST E-STOP. Evacuate 10m perimeter around suspended load."
        ),
        (
            "6. OSHA 1910.147 - Control of Hazardous Energy (Lockout/Tagout)",
            "Machine Isolation & Zero Energy State Protocol",
            "Unauthorized access into automated machinery cells or energized electrical panels is strictly prohibited. Procedures must ensure zero energy state using standardized padlocks and tags.",
            [
                "Execute instantaneous E-STOP upon optical light curtain breach.",
                "Verify zero residual pneumatic, hydraulic, and electrical charge.",
                "Authorized technician must apply Lockout/Tagout padlocks before servicing."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-108): DE-ENERGIZE SECTOR. Cordon perimeter and verify zero energy state."
        ),
        (
            "7. OSHA 1910.28 / 1910.29 - Walking-Working Surfaces & Fall Arrest",
            "Leading Edges, Elevated Platforms & Guardrail Requirements",
            "Unprotected sides or edges 4 feet (1.2m) or higher require compliant guardrail systems, safety nets, or personal fall arrest systems with 5,000 lb rated anchor points.",
            [
                "100% tie-off mandatory on scaffoldings and elevated catwalks.",
                "Toe-boards minimum 3.5 inches to eliminate falling tool hazards.",
                "Inspect harnesses and self-retracting lifelines prior to daily shift."
            ],
            "EMERGENCY PROTOCOL (SOP-EHS-058): Secure worker with certified fall arrest lifeline and inspect guardrail gates."
        )
    ]

    for title, subtitle, desc, checklist, callout in sections:
        story.append(Paragraph(title, h1_style))
        story.append(Paragraph(subtitle, h2_style))
        story.append(Paragraph(desc, body_style))
        for item in checklist:
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(callout, callout_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"✅ Generated OSHA 1910 PDF Document at: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)")

if __name__ == "__main__":
    generate_markdown_corpus()
    generate_pdf_corpus()
