import re
import spacy
from sentence_transformers import SentenceTransformer, util

# ===================== 配置 =====================
SEMANTIC_THRESHOLD = 0.50
SCENE_MARGIN = 0.03
FRICTION_THRESHOLD = 0.55
FRICTION_MARGIN = 0.05
MOTIVATION_THRESHOLD = 0.55
MOTIVATION_MARGIN = 0.03

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ===================== 加载模型 =====================
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

DOMAIN_KEYWORDS = {"ir", "fir", "led", "uv", "emf", "rf", "ems", "nmp", "red"}

# ===================== 规则和种子（请复制你本地的完整字典） =====================
MAIN_SCENE_MAP = {
    "fitness_recovery": "Physical Activity",
    "yoga": "Physical Activity",
    "stretching": "Physical Activity",
    "post_run": "Physical Activity",
    "post_cycling": "Physical Activity",
    "post_hiit": "Physical Activity",
    "during_workout": "Physical Activity",
    "couch_relax": "Home Lifestyle",
    "reading": "Home Lifestyle",
    "watching_tv": "Home Lifestyle",
    "gaming": "Home Lifestyle",
    "bedtime_relax": "Home Lifestyle",
    "nap": "Home Lifestyle",
    "winter_warm": "Home Lifestyle",
    "under_desk": "Work Context",
    "wfh": "Work Context",
    "long_hours_computer": "Work Context",
    "sedentary_recovery": "Work Context",
    "chronic_pain": "Wellness / Recovery",
    "nerve_pain": "Wellness / Recovery",
    "muscle_soreness": "Wellness / Recovery",
    "joint_recovery": "Wellness / Recovery",
    "back_recovery": "Wellness / Recovery",
    "surgery_recovery": "Wellness / Recovery",
    "sports_injury": "Wellness / Recovery",
    "nail_health": "Wellness / Recovery",
    "sleep_improvement": "Wellness / Recovery",
    "facial": "Beauty / Self-care",
    "skincare": "Beauty / Self-care",
    "wrinkle_care": "Beauty / Self-care",
    "acne": "Beauty / Self-care",
    "hair_growth": "Beauty / Self-care",
    "glow_complexion": "Beauty / Self-care",
    "body_firming": "Beauty / Self-care",
    "emotional_relaxing": "Emotional Narrative",
    "emotional_cozy": "Emotional Narrative",
    "emotional_calming": "Emotional Narrative",
    "emotional_comforting": "Emotional Narrative",
    "emotional_peaceful": "Emotional Narrative",
    "emotional_selfcare": "Emotional Narrative",
    "emotional_unwind": "Emotional Narrative",
}

SUB_PATTERNS = {
    "fitness_recovery": [
        r"\b(?:fitness\s+recovery|gym\s+recovery|post\s*workout\s+recovery|muscle\s+recovery\s+after\s+workout)\b",
        r"\bafter\s+workout\b",
    ],
    "yoga": [r"\byoga\b"],
    "stretching": [
        r"\b(?:stretch(?:ing)?\s+(?:routine|session|exercises?|every\s+morning|after\s+work)|do\s+(?:some\s+)?stretch(?:ing)?)\b"
    ],
    "post_run": [
        r"\b(?:after\s+running|post\s*run|running\s+recovery|recovery\s+after\s+run)\b"
    ],
    "post_cycling": [r"\b(?:after\s+cycling|post\s*cycling|cycling\s+recovery)\b"],
    "post_hiit": [r"\b(?:after\s+HIIT|post\s*HIIT|HIIT\s+recovery)\b"],
    "during_workout": [
        r"\b(?:use\s+(?:it|this)\s+(?:while|when|during)\s+(?:I\s+)?(?:work\s*out|do\s+(?:reps|cardio)|run|jog|cycle|lift|train)|"
        r"while\s+(?:working\s*out|exercising|doing\s+(?:reps|cardio|sets)|on\s+the\s+treadmill|using\s+the\s+(?:elliptical|bike|machine)))"
    ],
    "couch_relax": [
        r"\b(?:sofa|couch)\b.*\b(?:relax|chill|rest|lounge|kick\s*back|nap|unwind)\b",
        r"\b(?:relax|chill|lounge)\b.*\b(?:sofa|couch)\b",
    ],
    "reading": [
        r"\breading\s+(?:a\s+)?(?:book|novel|kindle|magazine)\b",
        r"\bread\s+(?:a\s+|my\s+)?(?:book|books|novel|kindle|magazine)\b",
    ],
    "watching_tv": [
        r"\b(?:watch(?:ing)?\s+(?:TV|television|netflix|hulu|movie)|binge\s*watch)\b"
    ],
    "gaming": [
        r"\b(?:play(?:ing)?\s+(?:video\s+)?game|gaming|console|xbox|playstation|nintendo)\b"
    ],
    "bedtime_relax": [
        r"\b(?:before\s+bed|bedtime|wind\s*down\s+(?:before\s+)?(?:sleep|bed)|nighttime\s+routine|fall\s+asleep)\b.*\b(?:relax|calm|cozy|comfort|unwind)\b",
        r"\b(?:relax|unwind)\b.*\b(?:before\s+bed|bedtime|sleep)\b",
        r"\bwind\s+down\b",
    ],
    "nap": [r"\b(?:nap|midday\s+sleep|afternoon\s+nap|siesta)\b"],
    "winter_warm": [
        r"\b(?:winter\s+warmth|keep\s+warm\s+in\s+winter|cold\s+weather\s+warm|heating\s+blanket)\b"
    ],
    "under_desk": [
        r"\b(?:under\s+(?:my\s+)?desk|footrest\s+(?:under\s+)?desk|desk\s+footrest)\b"
    ],
    "wfh": [
        r"\b(?:work\s+(?:from|at)\s+home|WFH|home\s*office|remote\s+work)\b",
        r"\b(?:in\s+my\s+office|at\s+the\s+office|office\s+use)\b",
        r"\b(?:doing\s+(?:my\s+)?(?:school\s+work|homework|assignments|studying)|while\s+studying)\b",
    ],
    "long_hours_computer": [
        r"\b(?:long\s+hours?\s+(?:at|in\s+front\s+of)\s+computer|working\s+on\s+computer\s+all\s+day|staring\s+at\s+screen\s+for\s+hours)\b"
    ],
    "sedentary_recovery": [
        r"\b(?:after\s+sitting\s+all\s+day|sedentary\s+(?:recovery|relief)|sitting\s+too\s+long)\b"
    ],
    "chronic_pain": [
        r"\b(?:chronic\s+pain|chronic\s+regional\s+pain\s+syndrome|CRPS|constant\s+pain|ongoing\s+pain|persistent\s+pain|stomach\s+pain|abdominal\s+pain|cramp(?:ing|s)?|bloating|digestive\s+(?:issue|problem|pain)|gut\s+pain|intestinal\s+pain|inflammation|skeletal\s+pain|migraine|headache)\b",
        r"\b(?:reliev(?:es?|ing)\s+(?:my\s+)?pain|pain\s+relief|relief\s+from\s+pain|eases?\s+(?:my\s+)?pain|help(?:s|ed)?\s+(?:with|my)\s+pain|pain\s+(?:is\s+)?gone|pain\s+(?:has\s+)?(?:reduced|lessened|decreased|diminished)|soothes?\s+pain)\b",
        r"\b(?:helps?\s+(?:me\s+)?with\s+my\s+(?:leg|arm|back|neck|foot|hand|knee|shoulder|hip|ankle|wrist|elbow|body|muscles?))\b",
        r"\b(?:uses?\s+(?:it|this|that|the\s+device|the\s+product)\s+on\s+(?:his|her|my|their|me)\s+(?:leg|arm|(?:lower\s+)?back|neck|foot|hand|knee|shoulder|hip|ankle|wrist|elbow|body|muscles?))\b",
        r"\b(?:us(?:e|ing)\s+(?:it|this|that|the\s+device|the\s+product)\s+(?:mostly\s+)?for\s+(?:my|his|her|their|me)\s+(?:leg|arm|back|neck|foot|hand|knee|shoulder|hip|ankle|wrist|elbow|body|muscles?))\b",
        r"\b(?:on|onto|upon)\s+(?:my|his|her|their|me)\s+(?:stomach|belly|tummy|abdomen|leg|arm|back|neck|foot|hand|knee|shoulder|hip|ankle|wrist|elbow|body|muscles?)\b",
        r"\b(?:help(?:s|ed)?\s+(?:my\s+)?(?:leg|arm|back|neck|foot|hand|knee|shoulder|hip|ankle|wrist|elbow|body|muscles?)(?:\s+(?:and|or)\s+(?:my\s+)?(?:leg|arm|back|neck|foot|hand|knee|shoulder|hip|ankle|wrist|elbow|body|muscles?))*\s+pain)\b",
        r"\b(?:provid(?:es?|ing)|gives?|offers?)\s+(?:some\s+)?(?:good\s+)?relief\b",
    ],
    "nerve_pain": [
        r"\b(?:neuropathy|nerve\s+pain|neuralgia|neuropathic\s+pain|nerve\s+damage|peripheral\s+neuropathy|diabetic\s+neuropathy)\b",
        r"\b(?:reliev(?:es?|ing)\s+(?:my\s+)?(?:nerve\s+pain|neuropathy)|treat(?:ing|ed|s)?\s+(?:my\s+)?(?:neuropathy|nerve\s+pain))\b",
    ],
    "muscle_soreness": [
        r"\b(?:muscle\s+soreness|sore\s+muscles|DOMS|delayed\s+onset\s+muscle)\b",
        r"\b(?:reliev(?:es?|ing)\s+(?:my\s+)?(?:muscle\s+)?(?:soreness|stiffness))\b",
        r"\b(?:eases?\s+(?:muscle\s+)?(?:soreness|stiffness))\b",
        r"\b(?:stiffness|soreness)\s+(?:has\s+been|is|was|became|got)\s+(?:minimal|gone|better|reduced|less|much\s+less|improved)\b",
    ],
    "joint_recovery": [
        r"\b(?:joint\s+(?:recovery|pain|relief|problem|issue)|arthritis|arthritic|rheumatoid|RA|osteoarthritis|knee\s+(?:recovery|pain|relief|injury|problem|issue|heal|surgery|replacement)|shoulder\s+(?:recovery|pain|relief|injury)|rotator\s+cuff|elbow\s+(?:recovery|pain|injury)|hip\s+(?:recovery|pain|relief|injury)|ankle\s+(?:recovery|pain|injury)|wrist\s+(?:recovery|pain|injury)|gout|healed\s+my\s+knee|fixed\s+my\s+knee|knee\s+injury)\b",
        r"\b(?:knee|shoulder|hip|ankle|elbow|wrist|joint)\s+(?:stiffness|soreness|ache|pain)\b",
        r"\b(?:stiffness|soreness|ache)\s+(?:in|of)\s+(?:my\s+)?(?:knee|joint|shoulder|hip|ankle|elbow|wrist)\b",
        r"\b(?:stiff|sore|aching|painful)\s+(?:joints?|knees?|shoulders?|hips?|ankles?|elbows?|wrists?)\b",
        r"\b(?:hurt|injured)\s+(?:his|her|my|their)\s+(?:knee|shoulder|hip|ankle|elbow|wrist)\b",
        r"\b(?:no\s+pain\s+(?:in|on)\s+(?:my\s+)?(?:knee|joint|shoulder|hip|ankle|elbow|wrist|knees?|joints?)|pain\s+free\s+(?:in|on)\s+(?:my\s+)?(?:knee|joint|shoulder|hip|ankle|elbow|wrist|knees?|joints?)|(?:my\s+)?(?:knee|joint|shoulder|hip|ankle|elbow|wrist|knees?|joints?)\s+(?:is|are|feel|feels)\s+pain\s+free)\b",
        r"\b(?:notic(?:e|ed|ing)|see(?:n|s)?|observ(?:e|ed|ing))\s+(?:some\s+)?(?:differences?|changes?|improvements?)\s+(?:in|on|with)\s+(?:my\s+)?(?:joints?|knees?|shoulders?|hips?|ankles?|elbows?|wrists?)\b",
    ],
    "back_recovery": [
        r"\b(?:lower\s+back\s+(?:pain|recovery|relief|problems?|issues?)|back\s+recovery|sciatica|back\s+ache|back\s+pain|strain(?:ed)?\s+back)\b",
        r"\b(?:back|spine|lower\s+back)\s+(?:stiffness|soreness|ache)\b",
        r"\b(?:stiffness|soreness|ache)\s+(?:in|of)\s+(?:my\s+)?(?:back|spine|lower\s+back)\b",
        r"\b(?:reliev(?:es?|ing)\s+(?:my\s+)?(?:back\s+)?(?:stiffness|soreness)|eases?\s+(?:my\s+)?(?:back\s+)?(?:stiffness|soreness))\b",
    ],
    "surgery_recovery": [
        r"\b(?:post[\s-]surgery|after\s+surgery|surgery\s+recovery|recovery\s+from\s+surgery|aid\s+in\s+recovery(?:/\w+)?\b|recover(?:ing|y)\s+from\s+(?:my\s+)?surgery|heal(?:ing)?\s+after\s+surgery)\b",
        r"\b(?:major\s+surgery|had\s+surgery|underwent\s+surgery|surgical\s+procedure)\b",
    ],
    "sports_injury": [
        r"\b(?:sports\s+injury|injured\s+(?:while|during)\s+(?:playing|running|training)|sprain(?:ed)?|strain\s+recovery)\b",
        r"\b(?:tendon(?:itis)?|ligament|tear|torn|reattach|rebuild|rupture|surgery\s+recovery|post\s*surgery|rehab(?:ilitation)?|physical\s+therapy)\b",
        r"\b(?:bicep\s+tendon|ACL|MCL|meniscus|achilles)\b",
    ],
    "nail_health": [
        r"\b(?:toenail\s*fungus|nail\s*fungus|fungal\s+infection\s+(?:on\s+)?(?:toe|foot|nail)|athlete\s*foot|foot\s*fungus)\b",
        r"\b(?:clear(?:ed|s|ing)?\s+up\s+(?:my\s+)?(?:toenail|nail|foot|fungus)|treat(?:ing|ed|s)?\s+(?:my\s+)?(?:toenail|nail|foot))\b",
    ],
    "sleep_improvement": [
        r"\b(?:improve(?:s|d)?\s+(?:my\s+)?(?:sleep\s+quality|sleep|quality\s+of\s+sleep)|better\s+sleep|sleep\s+better|help(?:s|ed)?\s+(?:me\s+)?sleep|insomnia|trouble\s+sleeping|hard\s+time\s+sleeping|sleep\s+issues?)\b"
    ],
    "facial": [
        r"\b(?:facial\s+(?:care|treatment|routine|mask|device|gadget|massager|toner|roller)|face\s+massager|face\s+roller|LED\s+mask)\b",
        r"\b(?:red\s+setting|blue\s+setting|LED\s+mode)\b",
    ],
    "skincare": [
        r"\b(?:skincare|skin\s*care|skin\s+(?:routine|regimen|changes?|improvement|texture|feels?\s+\w+|looks?\s+\w+|became\s+\w+|sensitive|redness|irritation|calm|soothe|smooth|glow))\b",
        r"\b(?:my\s+skin\s+(?:is|feels|became|got)\s+(?:sensitive|smooth|red|irritated|calm|clear|soft|healthy|better))\b",
        r"\b(?:no\s+redness|no\s+irritation|didn't\s+feel\s+hot)\b",
        r"\b(?:help(?:s|ed)?\s+(?:me\s+)?(?:a\s+lot\s+)?(?:with|for)\s+my\s+skin)\b",
        r"\b(?:notic(?:e|ed|ing)|see(?:n|s)?|observ(?:e|ed|ing))\s+(?:some\s+)?(?:differences?|changes?|improvements?)\s+(?:in|on|with)\s+(?:my\s+)?skin\b",
        r"\b(?:scar|scars|scarring)\b",
        r"\b(?:collagen|boost\s+collagen|collagen\s+stimulation|collagen\s+production)\b",
    ],
    "wrinkle_care": [
        r"\b(?:wrinkle|anti[\s-]*aging|skin\s+aging|fine\s+lines?|firming|mature\s+skin|age\s*spots|sagging\s+skin|youthful|rejuvenat)\b"
    ],
    "acne": [r"\b(?:acne|pimple|breakout|blemish|blackhead)\b"],
    "glow_complexion": [
        r"\b(?:glowing\s+(?:skin|body)|skin\s+glow|body\s+glow|glow(?:i(?:er|est))?\s+(?:skin|complexion|face|body)|radiant|complexion|dewy|brighten(?:ing)?\s+skin|look\s+(?:more\s+)?youthful|skin\s+(?:tone|soft|smooth|healthier|clearer))\b"
    ],
    "body_firming": [
        r"\b(?:skin\s+tightening|body\s+firming|tummy\s+firmer|butt\s+firmer|tighten(?:ing)?\s+(?:my\s+)?(?:skin|body|tummy|belly|thighs?|arms?|butt|stomach)|firm(?:er|ing)?\s+(?:my\s+)?(?:tummy|belly|butt|thighs?|arms?|skin))\b",
        r"\b(?:treat\s+(?:my\s+)?(?:belly|tummy|butt|thighs?|arms?|love\s*handles))\b",
    ],
    "hair_growth": [
        r"\b(?:hair\s*growth|hair\s+grow(?:ing|s|th)?|grow(?:ing)?\s+(?:my\s+)?hair|hair\s+loss|thinning\s+hair|bald(?:ing|ness)?|receding\s+hairline|alopecia|regrowth|new\s+hair|thicker\s+hair|hair\s+thicker|density|shedding|beard|facial\s+hair)\b",
        r"\b(?:made\s+my\s+hair\s+grow|stopped\s+my\s+hair\s+(?:loss|fall)|filled\s+in\s+(?:my\s+)?bald\s+spots?)\b",
    ],
    "emotional_relaxing": [r"\b(?:relaxing|relaxation)\b"],
    "emotional_cozy": [r"\bcozy\b"],
    "emotional_calming": [r"\bcalming\b"],
    "emotional_comforting": [r"\bcomforting\b"],
    "emotional_peaceful": [r"\bpeaceful\b"],
    "emotional_selfcare": [r"\bself[\s-]*care\b"],
    "emotional_unwind": [r"\bunwind\b"],
}

FRICTION_TAXONOMY = {
    "Physical": {
        "Sensory_Overload": [
            r"\b(?:too\s+bright|hurts?\s+(?:my\s+)?eyes?|need\s+goggles|makes?\s+me\s+dizzy|light\s+is\s+intense|blinding|eye\s+strain|strains?\s+(?:my\s+)?eyes?|sensitive\s+to\s+light|light\s+sensitivity)\b",
            r"\b(?:need\s+(?:to\s+wear\s+)?(?:goggles|protective\s+eyewear|eye\s+protection|the\s+provided\s+goggles)|"
            r"have\s+to\s+(?:wear|use)\s+(?:goggles|eye\s+protection|the\s+goggles)|"
            r"can't\s+use\s+without\s+(?:goggles|eye\s+protection)|"
            r"goggles\s+(?:are|is)\s+(?:required|necessary|a\s+must))\b",
            r"\b(?:IR\s+damage|temporarily\s+blinded|not\s+safe\s+for\s+eyes?|eye\s+damage|burned?\s+(?:my|the)\s+retina|color\s+distortion)\b",
            r"\b(?:feel(?:s)?\s+(?:too\s+)?intense|too\s+intense|intense\s+(?:glow|light))\b",
            r"\b(?:blinds?\s+(?:me|you|my\s+eyes)|is\s+blinding|blinding\s+light)\b",
        ],
        "Insufficient_Heat": [
            r"\b(?:does\s+not\s+heat|never\s+heats?\s+(?:up\s+)?(?:to|past|over)\s+\d+|doesn't\s+(?:get\s+)?hot\s+enough|won't\s+(?:heat|reach|get\s+to)\s+\d+|not\s+heating\s+properly|doesn't\s+heat\s+at\s+all|"
            r"power\s+(?:is|was|feels?)\s+too\s+(?:low|weak|feeble)|temperature\s+(?:is|was|rises?|warms?\s+up)\s+too\s+(?:slow|low|gradual)|(?:doesn't|won't)\s+get\s+(?:hot|warm)\s+enough|not\s+powerful\s+enough\s+to\s+(?:heat|make\s+a\s+difference)|"
            r"takes?\s+too\s+long\s+to\s+(?:heat|warm|get\s+hot)|heating\s+(?:is|seems?)\s+(?:weak|slow|insufficient))\b"
        ],
        "EMF_Radiation_Concern": [
            r"\b(?:assures?\s+(?:there\s+is\s+)?no\s+EMF\s+emission|EMF\s+detector\s+went\s+off|misrepresentation|falsely?\s+claim(?:s|ed|ing)?\s+(?:no|zero)\s+(?:EMF|radiation|emissions?)|"
            r"documentation\s+(?:says|assures?|claims?)\s+(?:no|zero)\s+(?:EMF|radiation|emissions?)\s+but\b|"
            r"advertised\s+as\s+(?:no|zero)\s+(?:EMF|radiation|emissions?)|"
            r"EMF\s+(?:radiation|emissions?)\s+(?:is|are)\s+(?:high|dangerous|unsafe|a\s+concern)|"
            r"concern(?:ed|ing)?\s+about\s+(?:EMF|radiation|electromagnetic))\b",
            r"\b(?:EMF\s+levels?\s+(?:is|are|was|were)\s+(?:through\s+the\s+roof|off\s+the\s+charts|extremely\s+high|dangerously\s+high)|"
            r"(?:low|zero|no)\s+EMF\s+(?:levels?|emissions?)\s+(?:as\s+advertised|claimed)\s+but)\b",
            r"\b(?:EMF\s+levels?\s+(?:are|is)\s+(?:way|far)\s+too\s+high)\b",
        ],
        "Stand_Adjustment_Issue": [
            r"\b(?:stand\s+(?:is\s+)?(?:very\s+)?(?:stiff|hard\s+to\s+(?:adjust|move|position|tilt|angle)|difficult\s+to\s+adjust|not\s+adjustable|won't\s+stay|doesn't\s+hold\s+position)|"
            r"adjust(?:ing|ment)?\s+(?:is\s+)?(?:stiff|hard|difficult|a\s+pain)|"
            r"(?:stiff|hard\s+to\s+adjust)\s+(?:stand|arm|joint|hinge|mount)|"
            r"can't\s+(?:adjust|move)\s+the\s+(?:stand|arm|panel))\b",
            r"\b(?:stand\s+(?:is\s+)?(?:unstable|wobbly|flimsy|poorly\s+balanced|not\s+balanced|will\s+fall\s+over|tips?\s+over|falls?\s+over)|"
            r"base\s+(?:is\s+)?(?:not\s+properly\s+balanced|unbalanced|too\s+light|wobbly)|"
            r"(?:stand|base)\s+(?:makes?\s+it\s+)?(?:easy\s+to\s+tip|tips?\s+over))\b",
            r"\b(?:comes?\s+apart\s+in\s+(?:my|your)\s+hand|threads?\s+(?:are\s+)?(?:starting\s+to\s+)?strip|stand\s+(?:fell|falls?|came)\s+apart|"
            r"(?:stand|arm|joint|hinge)\s+(?:is\s+)?(?:loose|won't\s+stay\s+tight|keeps?\s+loosening|flops?\s+around))\b",
            r"\b(?:height\s+(?:is|was|isn't|is\s+not)\s+(?:great|adjustable|sufficient|enough|too\s+(?:low|high|short|tall))|had\s+to\s+put\s+(?:a|something|a\s+wooden\s+crate)\s+under\s+(?:it|the\s+stand|the\s+lamp)|stand\s+(?:is\s+)?(?:too\s+short|too\s+low|not\s+tall\s+enough))\b",
        ],
        "Button_Sensitivity": [
            r"\b(?:buttons?\s+(?:is|are|weren't|aren't)\s+(?:too\s+)?(?:sensitive|easy\s+to\s+press|hard\s+to\s+avoid)|"
            r"accidentally\s+(?:hit|press|touch|trigger|click)\s+(?:the\s+)?(?:buttons?|timer|control|switch)|"
            r"keep\s+(?:accidentally\s+)?(?:hitting|pressing|touching)\s+(?:the\s+)?(?:buttons?|timer)|"
            r"buttons?\s+(?:are|is)\s+(?:way\s+)?too\s+(?:sensitive|easy\s+to\s+activate))\b"
        ],
        "Control_Placement_Issue": [
            r"\b(?:switch|control|button|display|settings?)\s+(?:should\s+(?:be|have\s+been)\s+)?(?:on\s+the\s+cord|on\s+the\s+handle|in\s+front|at\s+the\s+top|easier\s+to\s+(?:reach|see|use)|hard\s+to\s+(?:see|reach|find|use|operate|access)|awkward|unhandy|unhandled|inconvenient)|"
            r"(?:hard\s+to\s+see\s+(?:the\s+)?(?:settings?|display|controls?|buttons?)|very\s+unhand(?:led|y)|would\s+buy\s+one\s+with\s+control\s+on\s+cord)\b"
        ],
        "Cord_Adapter_Issue": [
            r"\b(?:cord\s+(?:is\s+)?(?:too\s+)?(?:short|bulky|annoying|in\s+the\s+way|a\s+problem|a\s+pain)|"
            r"adapter\s+(?:is\s+)?(?:bulky|large|heavy|in\s+the\s+way|awkward|a\s+hassle)|"
            r"(?:short|bulky)\s+(?:cord|adapter|power\s+cord)|"
            r"(?:cord|adapter)\s+makes?\s+(?:it|the\s+device)\s+(?:hard|difficult|awkward|a\s+hassle)\s+to\s+use)\b",
            r"\b(?:cord\s+(?:being|is|was)\s+on\s+the\s+top|cord\s+(?:placement|position)\s+(?:is\s+)?(?:awkward|bad|inconvenient|limits?\s+(?:movement|motion|range))|limits?\s+(?:its|the)\s+(?:movement|motion|range))",
        ],
        "LED_Density_Insufficient": [
            r"\b(?:LED\s+density\s+(?:is\s+)?(?:low|too\s+low|insufficient|poor|not\s+great)|"
            r"number\s+of\s+LEDs\s+(?:is\s+)?(?:the\s+)?same\s+(?:despite|but|on\s+a)\s+(?:larger|bigger)\s+(?:size|mat|panel|area)|"
            r"LEDs?\s+(?:are|is)\s+(?:too\s+)?(?:sparse|spread\s+out|few|underpowered)|"
            r"light\s+density\s+(?:is\s+)?(?:low|reduced|poor|not\s+sufficient)|"
            r"(?:fewer|less)\s+LEDs?\s+(?:per|than|despite|in))\b"
        ],
        "Adverse_Reaction": [
            r"\b(?:made\s+my\s+skin\s+worse|brought\s+(?:it|the\s+spots?)\s+back|dark\s+spots?\s+return(?:ed)?|"
            r"caused\s+(?:more\s+)?(?:pigmentation|freckles|sun\s*spots|melasma|breakouts?|acne|rash|irritation|burns?|hyperpigmentation)|"
            r"(?:left|gave)\s+me\s+(?:a\s+)?(?:burn|scar|mark|dark\s+spot|hyperpigmentation)|"
            r"(?:made|left)\s+my\s+skin\s+(?:worse|damaged|uneven|darker|hyperpigmented)|"
            r"ruined\s+my\s+(?:skin|face|complexion)|"
            r"do\s+not\s+buy\s+(?:this\s+)?(?:scam|dangerous\s+product))\b",
            r"\b(?:broke\s+out|raised\s+(?:red\s+)?(?:lumps|bumps|hives)|allergic\s+reaction|dermatologist|doctor\s+(?:visit|shot|appointment)|steroid\s+shot|cortisol|urticaria|contact\s+dermatitis)\b",
            r"\b(?:made\s+my\s+blood\s+pressure\s+(?:drop|fall|go\s+(?:down|low))|blood\s+pressure\s+(?:dropped|fell)|(?:got|felt|became)\s+lightheaded|felt\s+(?:faint|dizzy|woozy)|passed\s+out|nearly\s+fainted)\b",
            r"\b(?:was\s+left\s+with\s+(?:dark\s+spots?|hyperpigmentation|marks?|scar(?:s|ring)?)|left\s+me\s+with\s+(?:dark\s+spots?|hyperpigmentation|marks?|scar(?:s|ring)?))",
            r"\b(?:made\s+(?:my|the)\s+(?:injury|pain|condition)\s+worse|injured\s+(?:my|the)\s+\w+\s+further|aggravated\s+(?:my|the)\s+(?:injury|pain|condition)|design\s+(?:caused|made|led\s+to)\s+(?:an\s+)?injury)\b",
            r"\b(?:pinched\s+(?:so\s+hard|hard\s+enough)\s+(?:it\s+)?broke\s+(?:the\s+)?skin|broke\s+the\s+skin|too\s+aggressive|far\s+too\s+aggressive|aggressive\s+massager|skin\s+broke|pierced\s+the\s+skin)\b",
            r"\b(?:rub(?:s|bed)?\s+(?:your|my|the)\s+skin\s+raw|skin\s+(?:is|was|got|became)\s+broken|rubbed\s+(?:my|the)\s+skin\s+raw)\b",
            r"\b(?:hurt(?:s|ing)?\s+(?:my|the|me|after|from)\s*(?:neck|shoulder|back|arm|leg|knee|hip|wrist|elbow|ankle|body)?|bruis(?:e|ed|ing|es)\s+(?:my|the|me|after|from)|(?:left|gave)\s+me\s+(?:a\s+)?(?:bruise|bruises)|woke?\s+up\s+bruised\s+and\s+hurting)\b",
        ],
        "Skin_Compatibility": [
            r"\b(?:not\s+for\s+(?:melanated|dark|black|brown|ethnic|tanned)\s+skin|not\s+suitable\s+for\s+(?:dark|black|brown)\s+skin|(?:caused|left|gave\s+me)\s+(?:black|dark|brown)\s+(?:dots?|spots?|marks?|patches?)|hyperpigmentation\s+on\s+(?:dark|black|brown)\s+skin|ruined\s+my\s+(?:melanated|dark)\s+skin)\b",
            r"\b(?:caution\s+(?:black|african|dark[\s-]skinned)\s+(?:women|people|men|users?)\s+(?:against|about|before)\s+using|not\s+suitable\s+for\s+(?:black|african|dark[\s-]skinned)\s+(?:women|people|men|users?))",
        ],
        "Build_Quality": [
            r"\b(?:poor\s+quality|bad\s+quality|low\s+quality|cheaply\s+made|flimsy|not\s+well\s+made|"
            r"did\s+not\s+like\s+(?:the\s+)?quality|quality\s+(?:is|was|seems?)\s+(?:poor|bad|terrible|awful|low|not\s+good)|"
            r"quality\s+issues?|disappointed\s+with\s+(?:the\s+)?quality|"
            r"materials?\s+(?:feel|are|is)\s+(?:cheap|poor|flimsy)|"
            r"not\s+(?:the\s+)?quality\s+I\s+expected|lack\s+of\s+quality)\b",
            r"\b(?:wasn'?t\s+built\s+well|isn'?t\s+built\s+well|not\s+built\s+well|poorly\s+built)\b",
            r"\b(?:dye\s+transfer|bled\s+onto|color\s+bleed|quality\s+control\s+(?:problem|issue))\b",
            r"\b(?:cover\s+(?:isn'?t|is\s+not|not)\s+(?:removable|washable)|not\s+(?:removable|washable)|can'?t\s+(?:remove|wash)\s+the\s+cover)\b",
        ],
        "Wearing_Discomfort": [
            r"\b(?:uncomfortable\s+to\s+wear|very\s+uncomfortable|extremely\s+uncomfortable|not\s+comfortable\s+(?:to\s+wear|at\s+all)|couldn't\s+find\s+(?:even\s+)?one\s+(?:comfortable\s+)?position|unbearable\s+to\s+wear|could\s+only\s+wear\s+it\s+for\s+\d+\s+seconds?|can't\s+wear\s+it\s+for\s+more\s+than\s+\d+\s+(?:seconds?|minutes?))\b",
            r"\b(?:sore\s+(?:and|or)\s+bruised|bruised\s+(?:after|from|by)|goggle\s+imprints|"
            r"left\s+(?:marks?|imprints?|indentations?)|pressure\s+(?:marks?|points?|sores?)|"
            r"nose\s+bridge\s+(?:is|was|feels?)\s+(?:sore|painful|bruised))\b",
            r"\bnot\s+comfortable\b",
            r"\bnot\s+enough\s+cushion\b",
            r"\bnot\s+(?:comfirtable|confortable)\b",
            r"\b(?:strangle(?:d|s)?|choking\s+(?:feeling|sensation)|felt\s+(?:like\s+I\s+was\s+being\s+strangled|strangled)|a\s+bit\s+strangled)\b",
            r"\buncomfortable\b",
            r"\b(?:constrictive|choking\s+sensation|too\s+constricting)\b",
            r"\b(?:jaw\s+pressure|pushing\s+my\s+jaw|pressed\s+against\s+(?:my\s+)?jaw|chin\s+pressure)\b",
            r"\blacks?\s+support\b",
        ],
        "Thermal_Discomfort": [
            r"\b(?:gets?\s+too\s+hot|overheat|makes?\s+me\s+sweat|feels?\s+hot\s+on\s+(?:my\s+)?(?:face|skin)|uncomfortably\s+warm|burning\s+sensation|too\s+warm|heat\s+is\s+too\s+much)\b",
            r"\b(?:hot\s+under(?:neath)?\s+(?:it|the\s+mask|the\s+device)|hot\s+while\s+(?:wearing|using))\b",
            r"\btoo\s+hot\b",
            r"\b(?:made\s+me\s+(?:so|really|too)\s+hot|so\s+hot\s+(?:I|that|it)|feels?\s+so\s+hot)\b",
            r"\b(?:sweats?|sweating)\s+(?:when|while|with\s+it\s+on)\b",
        ],
        "Breath_Moisture": [
            r"\b(?:breath(?:e|ing)\s+causes?\s+moisture|moisture\s+on\s+the\s+face|"
            r"fog(?:s|ging|ged)\s+up|condensation\s+(?:inside|on\s+the\s+mask)|"
            r"reminded\s+me\s+of\s+(?:the\s+)?covid\s+mask|like\s+a\s+covid\s+mask|"
            r"breathing\s+(?:makes|creates)\s+(?:moisture|dampness))\b"
        ],
        "Posture_Burden": [
            r"\b(?:hard\s+to\s+stand\s+there|neck\s+(?:gets?\s+)?(?:tired|pain|sore)|awkward\s+position|have\s+to\s+hold\s+(?:it|my\s+arm)|can't\s+relax\s+while\s+using|need\s+to\s+sit\s+in\s+an\s+awkward\s+position)\b"
        ],
        "Fit_Issue": [
            r"\b(?:doesn't\s+fit\s+(?:my|the|properly|well|right|correctly|my\s+(?:face|head|neck|arm|elbow|leg|knee|thigh|wrist|body))|"
            r"too\s+(?:loose|tight|big|small|large)\s+(?:for\s+(?:my|the|his|her)\s+(?:face|head|neck|arm|elbow|leg|knee|thigh|wrist|body)|around\s+(?:my|the)\s+(?:neck|arm|elbow))|"
            r"strap\s+(?:is\s+)?(?:too\s+)?(?:loose|tight|short|long|doesn't\s+hold)|"
            r"small\s+(?:head|face|neck|frame)\b.*\b(?:can't|doesn't|won't|hard\s+to)\s+(?:hold|fit|keep\s+it\s+on)|"
            r"can't\s+get\s+(?:it\s+)?(?:to\s+)?(?:fit|stay\s+on|seal|around\s+(?:my\s+)?(?:neck|arm|elbow))|"
            r"slides?\s+(?:off|down|around)|"
            r"not\s+(?:snug|secure|comfortable)\s+(?:enough|fit|on\s+(?:my\s+)?(?:neck|arm|elbow))|"
            r"(?:neck|elbow)\s+(?:fit|support|wrap|strap)\s+is\s+(?:poor|bad|uncomfortable)|"
            r"(?:doesn't|won't)\s+(?:wrap|fit|stay)\s+around\s+(?:my\s+)?(?:neck|elbow))\b",
            r"\b(?:does\s+not\s+fit\s+well|doesn't\s+fit\s+well|fits?\s+poorly|not\s+a\s+good\s+fit)\b",
            r"\b(?:size\s+(?:is|was)\s+(?:too\s+(?:large|big|small|little|narrow|wide)|a\s+(?:bit|little)\s+too\s+(?:large|big|small)|not\s+(?:big|large|small)\s+enough)|"
            r"needs?\s+(?:to\s+be|to\s+be\s+about|at\s+least)?\s*\d+\s*(?:to\s+\d+\s*)?(?:inches?|feet|cm)?\s+(?:wider|longer|taller|bigger|smaller|shorter)|"
            r"should\s+(?:be|have\s+been)\s+(?:bigger|larger|smaller|wider|longer|taller)|"
            r"(?:too|a\s+(?:bit|little)\s+too)\s+(?:large|big|small|little|narrow|wide)\s+(?:in\s+)?size)\b",
            r"\b(?:very\s+loose|(?:doesn't|didn't)\s+(?:give|offer|provide)\s+(?:any\s+)?support|no\s+support(?:\s+(?:for|at\s+all))?|not\s+supportive)\b",
            r"\btoo\s+(?:small|thick|narrow|wide|short|big|large)\b",
            r"\b(?:too\s+floppy|floppy)\b",
            r"\b(?:still\s+|way\s+|a\s+(?:bit|little)?\s+)?to\s+(?:small|big|large|thick|narrow|wide|short|long|floppy|loose|tight)\b",
            r"\b(?:much\s+(?:smaller|bigger|larger)\s+(?:than|then)\s+(?:shown|advertised|expected|pictured|it\s+(?:appears|looks))|smaller\s+(?:than|then)\s+(?:in\s+)?(?:the\s+)?(?:pictures?|ads?|photos?))\b",
            r"\b(?:not\s+(?:very\s+)?firm|lacks?\s+firmness|not\s+sturdy)\b",
            r"\b(?:not\s+(?:very\s+)?much\s+support|failed\s+to\s+provide\s+(?:enough\s+)?support|didn't\s+help\s+much|needs?\s+more\s+support)\b",
            r"\b(?:didn't\s+sit\s+(?:right|well|nice|properly)|doesn't\s+sit\s+(?:right|well|nice|properly)|not\s+sitting\s+(?:right|well|nice))\b",
            r"\b(?:velcro\s+(?:too\s+short|needs?\s+(?:to\s+be\s+)?longer|should\s+be\s+longer)|strap\s+(?:should\s+be\s+longer|needs?\s+(?:to\s+be\s+)?longer))\b",
            r"\bawkward\s+height\b",
            r"\btoo\s+soft\b",
            r"\b(?:flat|flattened|no\s+loft|no\s+thickness)\b",
            r"\b(?:does\s+not\s+hold\s+(?:your|my|the)\s+head|won'?t\s+hold\s+(?:your|my|the)\s+head|head\s+(?:is|was|falls?|fell)\s+(?:forward|down)|no\s+head\s+support)\b",
        ],
        "Eyewear_Incompatibility": [
            r"\b(?:can't\s+wear\s+(?:my\s+)?(?:eye\s*glasses|glasses|spectacles|contacts)\s+(?:while|when)\s+(?:wearing|using)\s+the\s+mask|can't\s+fit\s+glasses\s+under(?:neath)?\s+the\s+mask|not\s+compatible\s+with\s+glasses)\b"
        ],
        "Coverage_Insufficient": [
            r"\b(?:doesn't\s+cover|does\s+not\s+cover|not\s+covering|won't\s+cover|"
            r"misses?\s+(?:the|my)\s+(?:11'?s|crow(?:'s)?\s*feet|wrinkle\s+areas?)|"
            r"doesn't\s+treat\s+(?:the|my)\s+(?:11'?s|crow(?:'s)?\s*feet|wrinkles?)|"
            r"leaves?\s+out\s+(?:the|my)\s+(?:11'?s|crow(?:'s)?\s*feet)|"
            r"most\s+common\s+wrinkle\s+areas?\s+not\s+covered)\b"
        ],
        "Device_Handling": [
            r"\b(?!not\s+(?:too\s+)?)(?:heavy|hard\s+to\s+move|bulky\s+panel|difficult\s+to\s+lift|unwieldy|too\s+large\s+to\s+handle|hard\s+to\s+carry)\b",
            r"\b(?:clunky|on\s+the\s+heavier\s+side|a\s+bit\s+heavy|heavier\s+than\s+(?:I\s+)?(?:expected|like|want))",
            r"\b(?:not\s+practical\s+to\s+(?:move|use|position|hold|keep)|hard\s+to\s+(?:position|maneuver|keep\s+in\s+place|maintain\s+(?:the\s+)?(?:right|recommended|ideal)\s+distance|get\s+the\s+right\s+angle)|difficult\s+to\s+(?:move\s+around|adjust\s+position|keep\s+at\s+the\s+right\s+distance|aim)|inconvenient\s+to\s+(?:move|use|position|adjust)|awkward\s+to\s+(?:move|position|aim|hold))",
        ],
    },
    "Behavioral": {
        "Activation_Energy": [
            r"\b(?:too\s+much\s+hassle|takes?\s+effort\s+to\s+set\s*up|don't\s+always\s+feel\s+like\s+using|hassle\s+to\s+use|lazy\s+to\s+use|hard\s+to\s+get\s+started)\b",
            r"\b(?:a\s+pain\s+to\s+use|such\s+a\s+pain\s+to\s+use|annoying\s+to\s+use)\b",
        ],
        "Consistency_Burden": [
            r"\b(?:hard\s+to\s+keep\s+up\s+every\s+day|forget\s+to\s+use|stopped\s+after\s+(?:two\s+weeks|a\s+month)|not\s+using\s+as\s+often|fell\s+off\s+the\s+routine|can't\s+stay\s+consistent)\b"
        ],
        "Time_Burden": [
            r"\b(?:takes?\s+\d+\s+minutes?\s+(?:every|each)\s+(?:session|time|day)|too\s+time[\s-]consuming|time\s+commitment|don't\s+have\s+the\s+time|time\s+investment)\b"
        ],
        "Routine_Disruption": [
            r"\b(?:doesn't\s+fit\s+(?:my|into)\s+(?:schedule|routine)|need\s+to\s+make\s+time\s+for\s+it|disrupts?\s+(?:my|the)\s+(?:day|routine|flow)|hard\s+to\s+fit\s+in)\b"
        ],
    },
    "Cognitive": {
        "Protocol_Confusion": [
            r"\b(?:not\s+sure\s+how\s+(?:far|long|to\s+use)|instructions?\s+(?:are|is)\s+unclear|confus(?:ed|ing)\s+(?:about|on|how)|don't\s+know\s+the\s+right\s+(?:distance|settings?|time)|how\s+(?:long|far)\s+should\s+I)\b",
            r"\b(?:need\s+to\s+read\s+the\s+instructions\s+to\s+get\s+the\s+functionality\s+correct|have\s+to\s+read\s+the\s+manual\s+to\s+figure\s+out\s+each\s+mode|instructions?\s+(?:are|is)\s+(?:necessary|required)\s+to\s+understand\s+the\s+modes)\b",
        ],
        "Efficacy_Ambiguity": [
            r"\b(?:not\s+sure\s+(?:it's|if\s+it\s+is)\s+working|hard\s+to\s+tell\s+(?:any|a)\s+difference|maybe\s+placebo|haven't\s+noticed\s+(?:any|much)\s+(?:results?|improvement|change)|can't\s+tell\s+if\s+it's\s+doing\s+anything)\b",
            r"\b(?:it\s+(?:is\s+)?(?:not\s+working|doesn't\s+work|didn't\s+work)|not\s+seeing\s+(?:any\s+)?(?:results?|improvement|difference)|no\s+effect)\b",
            r"\b(?:does\s+not\s+work\s+on\s+(?:the|my|his|her)\s+(?:nose|chin|forehead|cheeks?|face)|not\s+effective\s+on\s+(?:the|my)\s+\w+)\b",
            r"\b(?:doesn't\s+work|did\s+nothing|no\s+results?|useless|ineffective|didn't\s+do\s+anything)\b",
            r"\b(?:still\s+having\s+(?:pain|symptoms?|issues?|sciatica?|back\s+pain)|does\s+not\s+work|didn't\s+work|not\s+effective|didn't\s+help)\b",
            r"\b(?:does\s+not\s+(?:relieve|resolve|solve|help|fix|address|cure)\s+(?:my\s+)?(?:pain|symptoms?|discomfort|soreness|issues?|problem)|"
            r"doesn't\s+(?:relieve|resolve|solve|help|fix|address|cure)\s+(?:my\s+)?(?:pain|symptoms?|discomfort|soreness|issues?|problem)|"
            r"not\s+(?:relieving|resolving|solving|helping|fixing|addressing|curing)\s+(?:my\s+)?(?:pain|symptoms?|issues?))\b",
            r"\bdid\s+not\s+work\b",
            r"\b(?:may\s+(?:or\s+may\s+)?not\s+work|might\s+not\s+work|may\s+not\s+work\s+for\s+me)\b",
            r"\b(?:doesn'?t\s+prevent|won'?t\s+prevent|fails?\s+to\s+prevent|does\s+not\s+prevent)\b",
            r"\b(?:not\s+much\s+(?:results?|relief|improvement|difference|effect)|very\s+little\s+(?:results?|relief|improvement|difference|effect)|minimal\s+(?:results?|relief|improvement|difference|effect))\b",
        ],
        "Information_Overload": [
            r"\b(?:too\s+many\s+settings?|confusing\s+specifications?|don't\s+understand\s+the\s+wavelengths?|overwhelmed\s+by\s+(?:the\s+)?options?|parameter\s+overload)\b"
        ],
        "Transparency_Concern": [
            r"\b(?:misleading|false\s+advertising|not\s+as\s+described|they\s+(?:don't|do\s+not)\s+(?:show|mention|disclose)|hidden\s+(?:change|upgrade)|upgraded\s+(?:without|and\s+didn't)\s+(?:notice|tell|say)|had\s+to\s+(?:write|contact|email)\s+the\s+company\s+to\s+find\s+out|not\s+enough\s+(?:info|information|reviews|data)\s+(?:out\s+there\s+)?(?:to\s+feel\s+comfortable|to\s+know\s+if\s+it\s+works?))\b",
            r"\b(?:not\s+(?:how|what)\s+(?:it\s+)?(?:was|is)\s+(?:described|advertised|shown)|not\s+as\s+(?:described|advertised|pictured|shown))\b",
            r"\b(?:not\s+as\s+adverti[sz]ed|not\s+as\s+advertised)\b",
        ],
        "Decision_Fatigue": [
            r"\b(?:not\s+sure\s+which\s+mode\s+to\s+use|too\s+many\s+options?|can't\s+decide\s+(?:which|what)\s+(?:mode|setting)|paralyzed\s+by\s+choices?)\b"
        ],
    },
    "Emotional": {
        "Feels_Like_Chore": [
            r"\b(?:feels?\s+like\s+(?:a\s+)?(?:chore|task|job)|another\s+thing\s+to\s+do|force\s+myself\s+to\s+use|don't\s+look\s+forward\s+to\s+using|dread\s+using|have\s+to\s+make\s+myself)\b"
        ],
        "Guilt_Loop": [
            r"\b(?:should\s+use\s+(?:it|this)\s+more|just\s+sitting\s+there|haven't\s+used\s+(?:it|this)\s+in\s+(?:weeks?|months?)|feel\s+guilty\s+(?:not\s+using|about\s+not)|collecting\s+dust|waste\s+of\s+money\s+sitting)\b",
            r"\b(?:dust\s+collector|closet\s+dust|will\s+be\s+a\s+dust\s+collector|won't\s+use\s+it\s+as\s+much\s+as\s+expected|will\s+probably\s+not\s+use\s+it\s+as\s+much)\b",
        ],
        "Disappointment": [
            r"\b(?:expected\s+more|didn't\s+live\s+up\s+to\s+(?:the\s+)?hype|disappointed|disappointing|not\s+what\s+I\s+(?:expected|hoped|thought)|underwhelmed)\b",
            r"\b(?:horrible|terrible|awful)\b",
            r"\bunderwhelming\b",
            r"\bnot\s+what\s+I\s+was\s+(?:expecting|hoping|thinking)\b",
        ],
        "Anxiety": [
            r"\b(?:worried\s+about\s+(?:eye\s+damage|safety|side\s+effects?)|not\s+sure\s+if\s+(?:it's|this\s+is)\s+safe|anxious?\s+about\s+using|concerned\s+(?:about|it)\s+(?:might|could)\s+(?:damage|hurt|harm))\b"
        ],
    },
    "Environmental": {
        "Space_Occupancy": [
            r"\b(?!not\s+(?:too\s+)?)(?:takes?\s+up\s+(?:too\s+much\s+)?(?:room|space)|bulky|hard\s+to\s+find\s+(?:a\s+)?place\s+for|doesn't\s+fit\s+(?:in|on)\s+(?:my|the)\s+(?:desk|counter|shelf|room)|clutters?\s+(?:my|the)\s+room|footprint\s+is\s+(?:big|large))\b"
        ],
        "Aesthetic_Conflict": [
            r"\b(?:looks?\s+like\s+(?:a\s+)?(?:medical\s+(?:device|equipment)|heater|clinic)|doesn't\s+fit\s+(?:my|the)\s+(?:room|decor|aesthetic)|ugly|eyesore|not\s+aesthetically\s+pleasing)\b"
        ],
        "Household_Conflict": [
            r"\b(?:my\s+(?:wife|husband|partner|roommate|kids?)\s+(?:hates?|complains?|can't\s+stand)|too\s+bright\s+for\s+(?:my\s+)?(?:partner|spouse)|bothers?\s+(?:my|the)\s+(?:partner|family|dog|cat))\b"
        ],
    },
    "Lifecycle": {
        "Shipping_Damage": [
            r"\b(?:arrived\s+(?:broken|damaged|cracked)|package\s+was\s+damaged|came\s+in\s+poor\s+condition|shipping\s+damage)\b"
        ],
        "Regulatory_Approval_Issue": [
            r"\b(?:not\s+(?:FDA|FSA|HSA|HRA|medical)\s+(?:approved|cleared|registered|certified|eligible|covered|qualified|compatible|accepted)|"
            r"(?:doesn't|won't|cannot|can't)\s+(?:qualify|work|be\s+used)\s+(?:with|for|as)\s+(?:FDA|FSA|HSA|HRA|medical)|"
            r"(?:FDA|FSA|HSA|HRA)\s+(?:approval|clearance|registration|certification)\s+(?:is\s+)?(?:missing|not|absent|required|needed|necessary))\b"
        ],
        "Missing_Parts": [
            r"\b(?:missing\s+(?:all\s+)?(?:the\s+)?parts?|no\s+(?:hardware|mounting\s+kit|screws|brackets|instructions?|manual|remote|cord|cable|adapter)|parts?\s+(?:are|were|is)\s+missing|didn't\s+come\s+with\s+(?:the\s+)?(?:parts?|hardware|accessories)|incomplete\s+package|(?:came|arrived)\s+(?:with\s+)?parts?\s+missing)\b"
        ],
        "Setup_Complexity": [
            r"\b(?:assembly\s+took\s+forever|instructions?\s+were\s+terrible|hard\s+to\s+put\s+together|setup\s+(?:was|is)\s+(?:a\s+)?(?:nightmare|pain|difficult)|complicated\s+to\s+install)\b"
        ],
        "Registration_Hassle": [
            r"\b(?:register(?:ing|ed|ation)?\s+(?:is\s+)?(?:impossible|difficult|hard|a\s+nightmare|a\s+pain|a\s+hassle|broken|won't\s+work)|"
            r"can't\s+register|unable\s+to\s+register|trying\s+to\s+register\b.*\b(?:impossible|difficult|hassle|waste\s+of\s+time))\b"
        ],
        "Reliability_Issue": [
            r"\b(?:stopped\s+working\s+after\s+(?:a\s+)?(?:month|week|few\s+days)|power\s+supply\s+failed|defective|malfunction|broke\s+down|died\s+after)\b",
            r"\b(?:magnetic\s+connection\s+(?:can\s+leave\s+an\s+error|is\s+unstable)|connection\s+(?:issue|error|problem|must\s+be\s+kept\s+clean))\b",
            r"\b(?:no\s+longer\s+working|not\s+working\s+anymore|(?:stop|stops|stopped|stopping)\s+working|(?:quit|quits|quitted|quitting)\s+working)\b",
            r"\b(?:bulbs?|lights?|leds?|diodes?)\s+(?:went|gone|go)\s+out\b",
            r"\bnot\s+durable\b",
            r"\bno\s+longer\s+turns?\s+on\b",
            r"\bnot\s+sure\s+on\s+durability\b",
            r"\bnot\s+sure\s+how\s+durable\b",
            r"\b(?:does\s+not\s+heat|never\s+heats?\s+(?:up\s+)?(?:to|past|over)\s+\d+|doesn't\s+(?:get\s+)?hot\s+enough|won't\s+(?:heat|reach|get\s+to)\s+\d+|not\s+heating\s+properly|doesn't\s+heat\s+at\s+all)\b",
            r"\b(?:broke|broken)\s+after\s+(?:\d+\s+(?:use|uses|day|week|month|year)|one\s+use|first\s+use|a\s+few\s+(?:uses|days|weeks)|minimal\s+use)\b",
            r"\b(?:USB\s+(?:tip\s+|connector\s+)?(?:came|fell|broke|broken|apart|off)|connector\s+(?:came|fell|broke)\s+apart|USB\s+(?:tip\s+)?broke\s+off)\b",
            r"\b(?:lights?\s+(?:are|is|started|began)?\s*(?:flickering|flashing|blinking|dimming|failing)|flickering\s+(?:lights?|leds?)|(?:some|few|several)\s+(?:of\s+the\s+)?(?:lights?|leds?|bulbs?)\s+(?:are|were|have|started)?\s+(?:flickering|flashing|not\s+working|out))",
            r"\b(?:stop(?:s|ped|ping)?\s+working\s+after\s+(?:a\s+)?few\s+(?:months?|weeks?|days?|times?|uses?)|not\s+working\s+after\s+(?:a\s+)?few\s+(?:months?|weeks?|days?|times?|uses?))\b",
            r"\b(?:just\s+)?not\s+working\b",
            r"\b(?:broke\s+(?:at\s+)?(?:approximately|about|around)?\s*\d+\s*(?:years?|months?)\s*(?:old|in)?|lasted\s+only\s+(?:a\s+little\s+over\s+)?\d+\s*(?:years?|months?)|(?:only|just|barely)\s+lasted\s+\d+\s*(?:years?|months?))\b",
            r"\b(?:broken\s+(?:right\s+)?(?:out\s+of|off|from)\s+the\s+box|nothing\s+(?:was|is|worked|working|happened|happens)|DOA|dead\s+on\s+arrival|arrived\s+dead)\b",
        ],
        "Auto_Shutoff_Disruption": [
            r"\b(?:auto[\s-]shut[\s-]?off\s+(?:is\s+)?(?:only\s+)?\d+\s+(?:hours?|minutes?)\s+(?:max|maximum|long)|"
            r"wish\s+(?:it\s+)?(?:had|there\s+was)\s+(?:a\s+)?(?:\d+\s*(?:hours?|minutes?)\s+)?timer|"
            r"timer\s+(?:is\s+)?too\s+(?:short|quick)|"
            r"can't\s+(?:turn\s+off\s+)?auto[\s-]?shut[\s-]?off|"
            r"(?:woke|wakes?)\s+(?:me\s+)?up\s+(?:every\s+)?\d+\s+hours?\s+because\s+(?:it\s+)?(?:shut|turned)\s+off|"
            r"disrupt(?:s|ing|ed)?\s+(?:my\s+)?sleep\s+(?:because|due\s+to)\s+(?:the\s+)?auto[\s-]?shut[\s-]?off)\b",
            r"\b(?:shuts?\s+off\s+after\s+(?:like\s+)?\d+\s+minutes?|should\s+(?:stay\s+on|go|run)\s+until\s+I\s+turn\s+it\s+off)\b",
        ],
        "Material_Degradation": [
            r"\b(?:disintegrat(?:e|ing|ion)|shed(?:ding|s)?\s+(?:particles|fibers|material|skin|bits)|"
            r"flak(?:e|ing|es|ed)?\s+off|peel(?:ing|ed|s)?\s+off|"
            r"fall(?:ing|s)?\s+apart|crumbling|"
            r"material\s+(?:breakdown|degradation|deterioration)|worn\s+out\s+(?:fabric|cover|material)|"
            r"become\s+(?:brittle|sticky|messy)|unusable\s+due\s+to\s+wear)\b"
        ],
        "Battery_Life": [
            r"\b(?:battery\s+(?:life|drain|issues?|problem|sucks|terrible|bad|poor|awful|dies?|doesn't\s+last)|"
            r"charge\s+(?:doesn't\s+last|runs?\s+out\s+fast)|dies?\s+(?:within|in\s+)\d+\s+(?:minutes?|hours?|days?))\b"
        ],
        "Charging_Issue": [
            r"\b(?:not\s+charging|won't\s+charge|doesn't\s+charge|stops?\s+charging|charging\s+(?:error|issue|problem|failed|stopped)|blinks?\s+\d{2,3}\b|error\s+code\s*E?\d{1,2}\b|says?\s+E\d{1,2}|E\d{1,2}\s+error)\b",
            r"\b(?:won't\s+recharge|not\s+recharging|doesn't\s+recharge)\b",
        ],
        "Customer_Support": [
            r"\b(?:support\s+never\s+replied|warranty\s+process\s+was\s+painful|customer\s+service\s+(?:is|was)\s+(?:terrible|bad|unhelpful)|no\s+response\s+from\s+(?:the\s+)?(?:company|seller|manufacturer))\b",
            r"\b(?:worst\s+customer\s+(?:service|support|experience)|terrible\s+customer\s+(?:service|support)|customer\s+(?:service|support)\s+(?:is|was)\s+(?:the\s+)?(?:worst|terrible|awful))\b",
            r"\b(?:wanted\s+picture\s+after\s+picture|kept\s+asking\s+for\s+(?:more\s+)?(?:photos?|pictures?|documents?|information)|excessive\s+(?:photo|document)\s+requests?|felt\s+(?:a\s+)?little\s+strange\s+(?:so\s+I\s+(?:decided|just)\s+to\s+deal\s+with\s+it\s+myself)|customer\s+service\s+(?:was|is|felt)\s+(?:sketchy|off|strange|weird)|too\s+many\s+hoops\s+to\s+jump\s+through\s+with\s+customer\s+service)\b",
            r"\b(?:company\s+is\s+not\s+honest|dishonest\s+company|(?:given|provided)\s+several\s+different\s+addresses|conflicting\s+return\s+instructions?|had\s+to\s+email\s+back\s+and\s+forth\s+for\s+a\s+refund)\b",
            r"\b(?:not\s+recognized\s+by\s+(?:the\s+)?website|couldn'?t\s+(?:contact|reach)\s+customer\s+service|unable\s+to\s+contact)\b",
        ],
        "Spam_Harassment": [
            r"\b(?:spam\s+(?:text|message|email|calls?)|receiv(?:e|ing|ed)\s+spam|get\s+spam\s+(?:text|message|email)s?|unwanted\s+(?:text|message|email|calls?)|cannot\s+stop\s+(?:the\s+)?(?:spam|messages?|texts?)|different\s+phone\s+number\s+(?:each|every)\s+time|unsubscribe\s+(?:doesn't\s+work|can't\s+stop)|text\s+spam\s+from\s+(?:this\s+)?company|harass(?:ing|ed|ment)\s+(?:with\s+)?(?:texts?|messages?|calls?))\b"
        ],
        "Warranty_Deception": [
            r"\b(?:not\s+sure\s+the\s+warranty\s+exists|warranty\s+(?:doesn't\s+exist|is\s+(?:fake|a\s+scam|useless|pointless)|never\s+(?:got|received|activated))|scan\s+(?:a\s+code|the\s+QR)\s+to\s+unlock\s+(?:the\s+)?warranty|unlock\s+warranty\s+but\s+only\s+got\s+promotions|registered\s+(?:for|the)\s+warranty\s+but\s+(?:nothing|only\s+spam|no\s+confirmation)|warranty\s+registration\s+(?:is\s+)?(?:a\s+)?(?:scam|trick|lie))\b",
            r"\b(?:messed\s+up\s+my\s+phone\s+(?:it\s+was\s+some\s+ridiculous\s+app|after\s+scanning)|QR\s+code\s+(?:messed\s+up|damaged|broke)\s+my\s+phone|scanned\s+the\s+QR\s+code\s+and\s+it\s+(?:wanted|asked\s+for)\s+(?:a\s+)?credit\s+card|wanted\s+a\s+credit\s+card\s+number\s+after\s+scanning)\b",
            r"\b(?:QR\s+code\s+for\s+(?:warranty|directions|manual)\s+(?:wanted|asked|required)\s+(?:credit\s+card|payment|personal\s+info)|do\s+not\s+use\s+the\s+QR\s+code)\b",
        ],
        "Return_Friction": [
            r"\b(?:return\s+policy\s+(?:is|was)\s+(?:terrible|awful|bad|horrible|a\s+joke)|"
            r"read\s+(?:the\s+)?return\s+policy\s+(?:terms|conditions|fine\s+print)|"
            r"return\s+(?:terms|conditions)\s+(?:are|is|were)\s+(?:terrible|unfair|a\s+scam))\b",
            r"\b(?:not\s+refundable|non[\s-]refundable|no\s+refunds?|cannot\s+return|can't\s+return|stuck\s+with\s+(?:it|this)|under\s+the\s+bed\s+it\s+goes|waste\s+of\s+money\s+can't\s+return)\b",
            r"\b(?:haven'?t\s+received\s+(?:my|the|a)\s+refund|refund\s+(?:never|still\s+not|hasn't|haven't)\s+(?:arrived|received|processed|showed)|out\s+\$\d+\s+to\s+mail\s+it\s+back|(?:had|cost)\s+to\s+pay\s+return\s+shipping|return\s+shipping\s+(?:cost|fee|was|is)\s+\$\d+|not\s+refunded\s+(?:the|my|for)\s+(?:return\s+)?shipping)\b",
            r"\b(?:not\s+refundable|non[\s-]refundable|no\s+refunds?|cannot\s+return|can't\s+return|stuck\s+with\s+(?:it|this|the\s+(?:product|item|thing|device|unit))|under\s+the\s+bed\s+it\s+goes|waste\s+of\s+money\s+can't\s+return)\b",
        ],
    },
    "Economic": {
        "Price_Shock": [
            r"\b(?:too\s+(?:expensive|pricey|costly)|overpriced|out\s+of\s+(?:my\s+)?budget|can't\s+afford|steep\s+price|price\s+is\s+(?:high|steep))\b",
            r"\b(?:is\s+(?:too\s+)?expensive|so\s+expensive|really\s+expensive|very\s+expensive)\b",
            r"\b(?:price\s+(?:is\s+)?way\s+too\s+high|paid\s+way\s+too\s+much|price\s+is\s+insane|cost\s+way\s+too\s+much)\b",
            r"\b(?:price\s+is\s+(?:insane|crazy|nuts|beyond\s+understanding|unbelievable|outrageous)|cost\s+is\s+(?:insane|crazy|nuts|beyond\s+understanding))\b",
            r"\bpaid\s+too\s+much\b",
        ],
        "Value_Doubt": [
            r"\b(?:not\s+worth\s+the\s+(?:money|price|cost)|expected\s+better\s+for\s+the\s+price|waste\s+of\s+money|overpriced\s+for\s+what\s+you\s+get|not\s+a\s+good\s+value)\b",
            r"\b(?:don't\s+waste\s+(?:your|the)\s+money|waste\s+of\s+(?:your\s+)?money|money\s+down\s+the\s+drain)\b",
            r"\bnot\s+worth\s+the\s+premium\s+price\b",
            r"\bnot\s+worth\s+\$\d+\b",
            r"\bnot\s+(?:even\s+)?(?:remotely\s+)?worth\s+the\s+(?:money|price|cost)\b",
        ],
        "ROI_Uncertainty": [
            r"\b(?:not\s+sure\s+(?:it's|if\s+it\s+is)\s+worth\s+the\s+investment|will\s+it\s+pay\s+off|return\s+on\s+investment|expensive\s+gamble|might\s+not\s+be\s+worth\s+it)\b"
        ],
    },
    "Identity": {
        "Medical_Device_Identity": [
            r"\b(?:looks?\s+like\s+(?:a\s+)?medical\s+(?:device|equipment)|clinical\s+looking|makes?\s+(?:my\s+)?room\s+look\s+like\s+(?:a\s+)?(?:clinic|hospital)|feels?\s+like\s+(?:a\s+)?hospital\s+room|not\s+aesthetic)\b"
        ],
        "Category_Confusion": [
            r"\b(?:not\s+sure\s+what\s+this\s+thing\s+is|looks?\s+like\s+a\s+heater|confus(?:ed|ing)\s+(?:what|about\s+what)\s+(?:this|it)\s+is|what\s+even\s+is\s+this)\b"
        ],
    },
}

MOTIVATION_TAXONOMY = {
    "Health & Recovery": {
        "Back_Pain": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:back\s+pain|lower\s+back|sciatica)|purchased\s+for\s+back\s+relief|needed\s+something\s+for\s+my\s+back)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+(?:for|on|with)\s+(?:my\s+)?(?:back\s+pain|lower\s+back|sciatica)|"
            r"helps?\s+(?:with|relieve)\s+(?:my\s+)?(?:back\s+pain|sciatica|back\s+issues?)|"
            r"use\s+(?:it|this)\s+for\s+(?:my\s+)?(?:back|back\s+pain|sciatica)|"
            r"(?:great|effective|essential)\s+for\s+(?:treating|managing)\s+(?:back\s+pain|sciatica))\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:back\s+pain|lower\s+back|sciatica)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+treat|to\s+relieve)\s+(?:my\s+)?(?:back\s+pain|lower\s+back|sciatica)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+to\s+help\s+(?:ease|relieve|soothe|alleviate|reduce|treat|manage)\s+(?:my\s+)?(?:back\s+pain|lower\s+back|sciatica)\b",
        ],
        "Joint_Pain": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:joint\s+pain|knee\s+pain|arthritis|shoulder\s+pain|hip\s+pain)|purchased\s+for\s+joint\s+relief)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+(?:for|on|with)\s+(?:my\s+)?(?:joint\s+pain|arthritis|knee|shoulder|hip|elbow|wrist|ankle|gout)|"
            r"helps?\s+(?:with|relieve)\s+(?:my\s+)?(?:joint\s+pain|arthritis|knee|shoulder|hip|elbow|wrist|ankle|gout)|"
            r"use\s+(?:it|this)\s+for\s+(?:my\s+)?(?:joint\s+pain|arthritis|knee|shoulder|hip))",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:joint\s+pain|knee\s+pain|arthritis|shoulder\s+pain|hip\s+pain)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+treat|to\s+relieve)\s+(?:my\s+)?(?:joint\s+pain|arthritis|knee|shoulder|hip|elbow|wrist|ankle|gout)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+because\s+(?:it's|it\s+is)\s+(?:good|great|supposed\s+to\s+be\s+good)\s+for\s+(?:my\s+)?(?:joints?|joint\s+pain|arthritis|knee|shoulder|hip)\b",
        ],
        "Chronic_Pain": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?chronic\s+pain|managing\s+chronic\s+pain\s+with\s+this|chronic\s+pain\s+relief\s+is\s+why\s+I\s+bought)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+(?:for|on|with)\s+(?:my\s+)?(?:chronic\s+pain|nerve\s+pain|fibromyalgia|CRPS|neuropathy)|"
            r"helps?\s+(?:with|relieve)\s+(?:my\s+)?(?:chronic\s+pain|nerve\s+pain|fibromyalgia|CRPS|neuropathy)|"
            r"use\s+(?:it|this)\s+for\s+(?:my\s+)?(?:chronic\s+pain|nerve\s+pain))",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:chronic\s+pain|nerve\s+pain|fibromyalgia|CRPS|neuropathy)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+treat|to\s+relieve)\s+(?:my\s+)?(?:chronic\s+pain|nerve\s+pain|fibromyalgia|CRPS|neuropathy)\b",
        ],
        "Muscle_Recovery": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:muscle\s+recovery|sore\s+muscles|post[\s-]workout\s+recovery|muscle\s+pain)|needed\s+muscle\s+recovery\s+after\s+exercise)\b",
            r"\b(?:works?\s+(?:great|well)\s+(?:for|on)\s+(?:my\s+)?(?:muscle\s+recovery|sore\s+muscles|DOMS|post[\s-]workout|muscle\s+pain)|"
            r"helps?\s+(?:with|speed\s+up|relieve|reduce)\s+(?:my\s+)?(?:muscle\s+recovery|post[\s-]workout\s+recovery|muscle\s+pain|soreness|stiffness))",
            r"\b(?:helps?\s+(?:with|relieve|reduce)\s+(?:my\s+)?(?:muscle\s+)?(?:soreness|stiffness|tightness|knots|muscle\s+pain)|"
            r"(?:relieved|eased|helped)\s+(?:my\s+)?(?:soreness|stiffness|shoulder|neck|back|muscle\s+pain)\s+(?:pain|soreness)?)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:muscle\s+recovery|sore\s+muscles|post[\s-]workout\s+recovery|muscle\s+pain)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)?\s+(?:for|to\s+help\s+with|to\s+speed\s+up|to\s+relieve|to\s+reduce)\s+(?:my\s+)?(?:muscle\s+recovery|post[\s-]workout\s+recovery|soreness|stiffness|muscle\s+pain)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+because\s+(?:it's|it\s+is)\s+(?:good|great|supposed\s+to\s+be\s+good)\s+for\s+(?:my\s+)?(?:muscles?|muscle\s+recovery|sore\s+muscles|muscle\s+pain)\b",
            r"\b(?:will\s+be\s+using|gonna\s+use|plan\s+to\s+use)\s+(?:it|this)\s+for\s+(?:my\s+)?(?:sore\s+muscles?|muscle\s+recovery|muscle\s+pain|post[\s-]workout\s+recovery)\b",
        ],
        "Surgery_Recovery": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:surgery\s+recovery|post[\s-]surgery|recovery\s+from\s+surgery|after\s+surgery|healing\s+after\s+surgery|surgery\s+scar\s+healing|scar\s+healing\s+after\s+surgery|breast\s+reconstruction\s+recovery)|purchased\s+to\s+aid\s+recovery\s+after\s+surgery)\b",
            r"\b(?:to\s+help\s+with\s+(?:my\s+)?(?:surgery\s+scar\s+healing|post[\s-]surgical\s+recovery|recovery\s+after\s+(?:my\s+)?(?:breast\s+reconstruction|surgery)))\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:surgery\s+recovery|post[\s-]surgery|surgery\s+scar\s+healing)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)?\s+(?:for|to\s+help\s+with|to\s+speed\s+up)\s+(?:my\s+)?(?:surgery\s+recovery|post[\s-]surgery\s+healing|scar\s+healing\s+after\s+surgery)\b",
        ],
        "Exercise_Recovery": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:recovery\s+from\s+(?:running|cycling|hiit|gym)|sports\s+recovery)|athletic\s+recovery\s+is\s+why\s+I\s+purchased)\b",
            r"\b(?:works?\s+(?:great|well)\s+(?:for|on)\s+(?:my\s+)?(?:recovery\s+after\s+(?:running|cycling|gym|HIIT|training)|sports\s+recovery)|"
            r"helps?\s+(?:with|in)\s+(?:my\s+)?(?:running|cycling|gym|HIIT|training)\s+recovery)",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:recovery\s+from\s+(?:running|cycling|hiit|gym)|sports\s+recovery)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+improve)\s+(?:my\s+)?(?:recovery\s+after\s+(?:running|cycling|gym|HIIT|training)|sports\s+recovery)\b",
        ],
        "Energy_Boost": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:low\s+energy|fatigue|energy\s+levels?|tiredness)|needed\s+(?:an\s+)?energy\s+boost|more\s+energy\s+during\s+the\s+day)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+for\s+(?:my\s+)?(?:energy|fatigue|alertness)|helps?\s+(?:with|boost)\s+(?:my\s+)?energy)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:low\s+energy|fatigue|tiredness|energy\s+levels?)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+boost)\s+(?:my\s+)?(?:energy|alertness|mood)\b",
        ],
        "General_Pain_Relief": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:pain\s+relief|aches?\s+and\s+pains?|chronic\s+aches?|daily\s+pain)|needed\s+(?:some\s+)?pain\s+relief)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+for\s+(?:my\s+)?(?:pain\s+relief|aches?|soreness)|helps?\s+(?:relieve|ease)\s+(?:my\s+)?(?:pain|aches?|soreness)|reduc(?:es?|ing)\s+(?:my\s+)?pain)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:pain\s+relief|aches?\s+and\s+pains?|daily\s+pain)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+relieve)\s+(?:my\s+)?(?:pain|aches?|soreness|discomfort)\b",
            r"\b(?:got|gave\s+me|provided)\s+(?:almost\s+)?(?:immediate\s+)?pain\s+relief\b",
        ],
        "Sleep_Improvement": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:better\s+sleep|insomnia|sleep\s+quality|sleep\s+issues?)|wanted\s+to\s+sleep\s+better|sleep\s+was\s+the\s+main\s+reason)\b",
            r"\b(?:helps?\s+(?:me\s+)?(?:sleep|fall\s+asleep|insomnia)|works?\s+(?:great|well)\s+(?:for|with)\s+(?:my\s+)?(?:insomnia|sleep\s+issues?|trouble\s+sleeping)|"
            r"use\s+(?:it|this)\s+for\s+(?:my\s+)?(?:insomnia|sleep)|(?:improved|better)\s+(?:my\s+)?sleep)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:insomnia|sleep\s+issues?|trouble\s+sleeping)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+improve)\s+(?:my\s+)?(?:sleep|insomnia)\b",
            r"\b(?:helps?\s+with\s+(?:the\s+)?(?:quality|ease)\s+of\s+(?:getting\s+to\s+)?sleep)\b",
        ],
    },
    "Beauty & Self-Care": {
        "Anti_Aging": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:anti[\s-]aging|wrinkles?|fine\s+lines?|skin\s+aging)|purchased\s+to\s+look\s+younger|anti[\s-]aging\s+motivated\s+my\s+purchase)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+(?:for|on)\s+(?:my\s+)?(?:wrinkles?|fine\s+lines?|crow'?s?\s+feet|aging\s+skin|mature\s+skin)|"
            r"helps?\s+(?:with|reduce)\s+(?:my\s+)?(?:wrinkles?|fine\s+lines?|signs?\s+of\s+aging)|"
            r"use\s+(?:it|this)\s+for\s+(?:my\s+)?(?:wrinkles?|anti[\s-]aging|aging\s+skin))",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:wrinkles?|fine\s+lines?|crow'?s?\s+feet|aging\s+skin)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+reduce)\s+(?:my\s+)?(?:wrinkles?|fine\s+lines?|signs?\s+of\s+aging)\b",
        ],
        "Skin_Health": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+improve|because\s+of)\s+(?:my\s+)?(?:skin\s+health|skin\s+condition|acne|glow|complexion)|wanted\s+better\s+skin|skin\s+improvement\s+was\s+the\s+goal)\b",
            r"\b(?:works?\s+(?:great|well|wonders?)\s+(?:for|on)\s+(?:my\s+)?(?:acne|skin|breakouts?|blemishes|pimples|complexion|rosacea|scars|scarring)|"
            r"help(?:s|ing|ed)?\s+(?:with|clear|fade|reduce)\s+(?:my\s+)?(?:acne|breakouts?|skin|blemishes|complexion|scars|scarring|rosacea)|"
            r"use\s+(?:it|this)?\s+for\s+(?:my\s+)?(?:acne|skin|complexion|scars|scarring)|"
            r"(?:great|good|effective)\s+for\s+(?:acne\s+prone|oily|dry|sensitive)\s+skin)\b",
            r"\b(?:helps?\s+(?:with|reduce|fade)\s+(?:my\s+)?(?:redness|hyperpigmentation|dark\s+spots?|sun\s+spots?|uneven\s+skin\s+tone|texture|scars|scarring)|"
            r"reduc(?:ing|es?|ed)?\s+(?:redness|hyperpigmentation|dark\s+spots?|sun\s+spots?|scars|scarring)|"
            r"notice(?:d|ing)?\s+(?:a\s+)?reduction\s+in\s+(?:redness|hyperpigmentation|dark\s+spots?|scars|scarring))\b",
            r"\b(?:noticed\s+(?:fewer\s+(?:fine\s+lines?|wrinkles?)|more\s+even\s+tone|improved\s+texture)|"
            r"targets?\s+(?:texture|breakouts?|blemishes|acne|scars)\s+(?:perfectly|well|greatly)|"
            r"great\s+for\s+daily\s+skincare|helps?\s+(?:with|improve)\s+(?:skin\s+texture|uneven\s+tone|breakouts?))\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+improve|because\s+of)\s+(?:my\s+)?(?:skin\s+health|acne|glow|complexion|scars|scarring)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)?\s+(?:for|to\s+help\s+with|to\s+clear|to\s+improve|to\s+fade|to\s+reduce)\s+(?:my\s+)?(?:skin|acne|breakouts?|complexion|scars|scarring)\b",
            r"\b(?:am|is|are)\s+(?:using|trying)\s+(?:it|this)?\s+for\s+(?:my\s+)?(?:skin|acne|breakouts?|blemishes|pimples|complexion|rosacea|scars|scarring|redness|hyperpigmentation|dark\s+spots?|uneven\s+skin\s+tone|texture)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got|use|using|trying|started\s+using)\s+(?:it|this)?\s+(?:for|to\s+help\s+with|to\s+treat|to\s+clear|to\s+improve)\s+(?:my\s+)?(?:skin\s+conditions?|cold\s+sores|HSV\d?|herpes|eczema|psoriasis|dermatitis|fungal\s+skin)\b",
            r"\b(?:great|good|effective|excellent)\s+for\s+(?:skin\s+conditions?|cold\s+sores|HSV\d?|eczema|psoriasis|dermatitis)\b",
            r"\b(?:my\s+)?(?:hormonal\s+)?acne\s+(?:cleared|clears|clearing)\s+up\b",
            r"\b(?:to\s+help\s+(?:avoid|prevent|reduce)|avoid|prevent|reduce)\s+(?:my\s+)?(?:stretch\s+marks?)\b",
            r"\b(?:will\s+be\s+using|gonna\s+use|plan\s+to\s+use)\s+(?:it|this)\s+for\s+(?:my\s+)?(?:skin\s+improvement|skin\s+health|acne|glow|complexion|scars|scarring|redness|hyperpigmentation)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:mainly\s+)?to\s+improve\s+(?:my\s+)?skin\b",
            r"\b(?:effective\s+addition\s+to\s+(?:my\s+)?skincare|part\s+of\s+my\s+skincare\s+(?:plan|routine|regimen)|skincare\s+(?:plan|routine|regimen)\s+addition)\b",
            r"\bpart\s+of\s+my\s+(?:regular\s+)?skincare\s+(?:routine|plan|regimen)\b",
        ],
    },
    "Emotional Well-being": {
        "Stress_Reduction": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:stress\s+reduction|relaxation|unwind|destress)|needed\s+something\s+to\s+relax|stress\s+relief\s+motivated\s+me)\b",
            r"\b(?:helps?\s+(?:me\s+)?(?:relax|unwind|destress|calm\s+down)|works?\s+(?:great|well)\s+for\s+(?:stress\s+relief|relaxation|unwinding)|"
            r"use\s+(?:it|this)\s+(?:to\s+)?(?:relax|unwind|de[-]?stress))",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:stress|anxiety|nervousness|mental\s+health)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+reduce|to\s+calm)\s+(?:my\s+)?(?:stress|anxiety|tension)\b",
        ],
        "Mood_Improvement": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:mood|depression|anxiety|seasonal\s+affective\s+disorder|SAD|mental\s+health)|purchased\s+to\s+(?:lift|boost|improve)\s+my\s+mood)\b",
            r"\b(?:helps?\s+(?:with|improve)\s+(?:my\s+)?(?:mood|depression|anxiety|SAD|mental\s+clarity)|brightens?\s+my\s+mood|mood\s+boosting)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|to\s+help\s+with|because\s+of)\s+(?:my\s+)?(?:mood|depression|seasonal\s+affective\s+disorder|SAD)\b",
            r"\b(?:use|using|trying|started\s+using)\s+(?:it|this)\s+(?:for|to\s+help\s+with|to\s+lift|to\s+improve)\s+(?:my\s+)?(?:mood|mental\s+health)\b",
        ],
    },
    "Social Influence": {
        "Personal_Recommendation": [
            r"\b(?:my\s+(?:wife|husband|mom|dad|friends?|sisters?|brothers?|coworkers?|colleagues?|therapists?|PT|physical\s+therapists?|trainers?|chiropractors?)\s+(?:recommend(?:s|ed)?|suggest(?:s|ed)?|t(?:old|ells?)\s+me\s+about|insist(?:s|ed)?|push(?:es|ed)?\s+me\s+to\s+get))|"
            r"\bfriends?\s+of\s+mine\s+recommend(?:s|ed)?|family\s+members?\s+recommend(?:s|ed)?\b",
            r"\b(?:my\s+)?(?:coworkers?|colleagues?|friends?|wife|husband|mom|dad|sisters?|brothers?)\s+t(?:old|ells?)\s+me\s+to\s+(?:check\s+out|look\s+into|try)\b",
        ],
        "Professional_Recommendation": [
            r"\b(?:my\s+(?:\w+\s+)?(?:doctor(?:'s)?|doctors?|physician(?:'s)?|physicians?|clinician(?:'s)?|clinicians?|dermatologist(?:'s)?|dermatologists?|specialist(?:'s)?|specialists?)\s+(?:recommend(?:s|ed)?|suggest(?:s|ed)?|prescrib(?:es?|ed)|t(?:old|ells?)\s+me\s+to\s+get)|"
            r"\b(?:by|on|following)\s+(?:the\s+)?recommendation\s+of\s+(?:my|me|the|a)\s+(?:\w+\s+)?(?:doctor(?:'s)?|doctors?|physician(?:'s)?|physicians?|clinician(?:'s)?|clinicians?|dermatologist(?:'s)?|dermatologists?|specialist(?:'s)?|specialists?)|"
            r"\b(?:this|that|the|a)\s+brand\s+was\s+recommended\s+by\s+(?:the|a|my|me)\s+(?:\w+\s+)?(?:doctor(?:'s)?|doctors?|physician(?:'s)?|physicians?|clinician(?:'s)?|clinicians?|dermatologist(?:'s)?|dermatologists?|specialist(?:'s)?|specialists?)|"
            r"\b(?:recommend(?:s|ed)?|suggest(?:s|ed)?|prescrib(?:es?|ed))\s+by\s+(?:my|me|the|a)\s+(?:\w+\s+)?(?:doctor(?:'s)?|doctors?|physician(?:'s)?|physicians?|clinician(?:'s)?|clinicians?|dermatologist(?:'s)?|dermatologists?|specialist(?:'s)?|specialists?)|"
            r"\b(?:the|a)\s+(?:\w+\s+)?(?:doctor|physician|clinician|dermatologist|specialist)\s+(?:s(?:ays|aid)|t(?:old|ells?)\s+me|mention(?:s|ed)?|recommend(?:s|ed)?|suggest(?:s|ed)?)\b|"
            r"professional\s+recommendation)\b"
        ],
        "Online_Reviews": [
            r"\b(?:read\s+(?:the\s+)?(?:good|great|positive|reviews?|ratings?)\s+and\s+decided|Amazon\s+reviews?\s+convinced\s+me|reviews?\s+were\s+(?:so\s+)?(?:good|persuasive|convincing))\b"
        ],
        "Influencer_Media": [
            r"\b(?:saw\s+(?:it|this)\s+on\s+(?:YouTube|TikTok|Instagram|Facebook|a\s+blog|a\s+review\s+channel)|influencer\s+recommended|YouTuber\s+review|social\s+media\s+post\s+about)\b"
        ],
    },
    "External Triggers": {
        "Discount_Used": [
            r"\b(?:bought\s+(?:it|this|one)\s+because\s+(?:it\s+was\s+)?(?:on\s+sale|discounted|a\s+deal)|"
            r"used\s+(?:a\s+)?(?:coupon|code|promo|voucher)|"
            r"lightning\s+deal|prime\s+day|"
            r"deal\s+of\s+the\s+day|"
            r"bought\s+on\s+black\s+friday|black\s+friday\s+(?:deal|sale|purchase)|"
            r"on\s+sale\s+and\s+I\s+(?:had\s+to\s+|decided\s+to\s+)?(?:get|buy|purchase))\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:because|since|as)\s+(?:it\s+was\s+)?(?:on\s+sale|discounted|a\s+deal|cheap)\b",
        ],
        "Gift_Purchase": [
            r"\b(?:bought\s+(?:this|it|one)\s+(?:as\s+(?:a\s+)?gift|for\s+(?:a\s+)?gift)|"
            r"gift\s+for\s+(?:my\s+)?(?:wife|husband|son|daughter|friend|mom|dad|kid|partner)|"
            r"present\s+for|birthday\s+gift|christmas\s+gift|anniversary\s+gift)\b",
            r"\b(?:bought|purchased)\s+(?:this|it|one)\s+for\s+(?:my\s+)?(?:wife|husband|mom|dad|son|daughter|friend|partner|sister|brother)\b",
            r"\b(?:bought|ended\s+up\s+buying)\s+(?:two|another)\s+to\s+give\s+(?:as\s+(?:a\s+)?gift|away|to\s+a\s+friend)\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:for|as)\s+(?:a\s+)?(?:gift|present)\s+(?:for\s+)?(?:my\s+)?(?:wife|husband|mom|dad|son|daughter|friend|partner)\b",
            r"\b(?:bought|purchased)\s+(?:this|it|one|the\s+\w+)\s+(?:\w+\s+){0,20}?for\s+(?:my\s+)?(?:wife|husband|mom|dad|son|daughter|friend|partner|sister|brother)\b",
            r"\b(?:best|great|wonderful|amazing|fantastic|perfect)\s+gifts?\s+(?:my\s+(?:husband|wife|mom|dad|son|daughter|friend|partner|sister|brother)\s+(?:has\s+)?(?:ever\s+)?(?:gave|given|got)\s+(?:to\s+)?me)\b",
        ],
    },
    "Functional Needs": {
        "Replacement_Needed": [
            r"\b(?:replac(?:e|ing)\s+(?:my|the|old|broken|worn[\s-]out)|upgrade\s+from|needed\s+a\s+new\s+one|old\s+one\s+(?:broke|died|stopped\s+working))\b",
            r"\b(?:buy(?:ing)?|buys|bought|purchas(?:e|ing|ed)|order(?:ing|ed)?|got)\s+(?:this|it|one)\s+(?:to\s+replace|because\s+my\s+old\s+one\s+(?:broke|died|stopped\s+working)|as\s+a\s+replacement)\b",
        ],
    },
}

SCENE_SEEDS = {
    "fitness_recovery": [
        "I use this for muscle recovery after gym workouts",
        "Perfect for post-workout recovery at the fitness center",
        "This helps my muscles recover after intense training",
    ],
    "yoga": [
        "I use this during my yoga sessions",
        "Great for yoga practice at the studio",
        "Perfect for stretching during yoga class",
    ],
    "stretching": [
        "I use this for my daily stretching routine",
        "This helps me stretch after sitting all day",
        "Great for stretching exercises at home",
    ],
    "post_run": [
        "I apply this after my morning run",
        "Recovery after running is easier with this",
        "Perfect for post-run muscle relief",
    ],
    "post_cycling": [
        "I use this after cycling long distances",
        "Great for recovery after a bike ride",
        "Helps my legs recover after cycling",
    ],
    "post_hiit": [
        "Using this after HIIT workouts reduces soreness",
        "Perfect for recovery after high intensity interval training",
        "I rely on this after my HIIT sessions",
    ],
    "during_workout": [
        "I use this during my workouts and it helps me perform better.",
        "Wearing it while I do my reps makes a noticeable difference.",
        "Perfect for using on the treadmill, doesn't restrict movement at all.",
    ],
    "couch_relax": [
        "I use this while relaxing on the couch",
        "Perfect for lounging on the sofa in the evening",
        "I love using this when I chill on the sofa",
    ],
    "reading": [
        "I use this while reading a book in my armchair",
        "Perfect companion for my evening reading routine",
        "I always have this when I read my novel",
    ],
    "watching_tv": [
        "I use this while watching TV in the living room",
        "Great for binge-watching Netflix on the couch",
        "Perfect for a movie night at home",
    ],
    "gaming": [
        "I use this during long gaming sessions",
        "Great while playing video games on my console",
        "Perfect for gaming on the couch",
    ],
    "bedtime_relax": [
        "I use this before bed to wind down",
        "This helps me relax and fall asleep",
        "My bedtime routine includes this for relaxation",
    ],
    "nap": [
        "I use this during my afternoon nap",
        "Perfect for a quick midday nap",
        "Helps me fall asleep for a nap",
    ],
    "winter_warm": [
        "This keeps me warm during cold winter nights",
        "Perfect for staying cozy in winter",
        "I use this to stay warm when it's freezing outside",
    ],
    "under_desk": [
        "I use this under my desk at work",
        "Perfect as an under-desk footrest in the office",
        "Fits perfectly under my standing desk",
    ],
    "wfh": [
        "I use this in my home office while working from home",
        "Essential for my work-from-home setup",
        "Perfect for remote work at my desk",
    ],
    "long_hours_computer": [
        "I sit at my computer for long hours and this helps",
        "Working on a computer all day, this relieves my back",
        "I use this during marathon computer sessions",
    ],
    "sedentary_recovery": [
        "After sitting all day at the office, I use this to recover",
        "This helps with stiffness from a sedentary lifestyle",
        "Perfect after a long day of sitting",
    ],
    "chronic_pain": [
        "I suffer from chronic back pain and this brings relief",
        "Managing chronic pain with this daily",
        "Chronic pain relief is the main reason I bought this",
    ],
    "nerve_pain": [
        "I use this for my neuropathy pain in my feet",
        "This helps with my nerve pain after chemotherapy",
        "Peripheral neuropathy relief has been great with this device",
    ],
    "muscle_soreness": [
        "This relieves my muscle soreness after the gym",
        "Sore muscles feel better after using this",
        "Great for DOMS and post-workout soreness",
    ],
    "joint_recovery": [
        "I use this for my knee joint recovery",
        "Helps with shoulder joint pain after exercise",
        "Joint recovery supplement that really works",
        "My wife has rheumatoid arthritis and this eases her joint pain",
        "I have arthritis in my knees and this provides great relief",
    ],
    "back_recovery": [
        "This helps my lower back recover after injury",
        "Back recovery is much faster with this device",
        "Sciatica relief and back recovery in one",
    ],
    "surgery_recovery": [
        "I use this to aid in my recovery after surgery",
        "Post-surgery recovery has been faster with this device",
        "Helps with healing after a major surgery",
    ],
    "sports_injury": [
        "After my sports injury, this was recommended",
        "Helps me recover from a running injury",
        "Sprain recovery aided by this product",
        "I tore my bicep tendon and this helps with rebuilding",
        "Post-surgery rehabilitation for my knee injury",
    ],
    "nail_health": [
        "This cleared up my toenail fungus after a few weeks",
        "Using this for a fungal infection on my toenail",
        "Finally something that works on nail fungus",
    ],
    "sleep_improvement": [
        "This device improved my sleep quality significantly",
        "I use it to help me sleep better",
        "My sleep has improved since I started using this",
    ],
    "facial": [
        "I use this as part of my facial care routine",
        "This facial roller reduces puffiness in the morning",
        "Perfect for at-home facial massage",
    ],
    "skincare": [
        "This is now a staple in my skincare routine",
        "I apply this every morning for better skin",
        "Skincare product that improved my complexion",
    ],
    "wrinkle_care": [
        "Using this to reduce fine lines and wrinkles",
        "Anti-aging cream that really firms the skin",
        "Noticeable reduction in crow's feet after a month",
    ],
    "acne": [
        "This cleared up my acne breakouts",
        "Best product for hormonal acne I've ever used",
        "Stops pimples before they even form",
    ],
    "glow_complexion": [
        "Gives my skin a natural glow",
        "My complexion looks radiant after using this serum",
        "Finally found something that makes my skin dewy",
    ],
    "body_firming": [
        "I use this for skin tightening on my stomach and thighs",
        "This product firms my belly and makes my skin tighter",
        "Great for body contouring and tightening loose skin",
    ],
    "hair_growth": [
        "This product made my hair grow faster",
        "I noticed new hair growth after using this",
        "My hair is thicker and fuller now",
        "It stopped my hair loss completely",
    ],
    "emotional_relaxing": [
        "Using this feels so relaxing after a long day",
        "This creates a deeply relaxing experience",
    ],
    "emotional_cozy": [
        "It makes me feel cozy and warm",
        "Cozy vibes whenever I use this",
    ],
    "emotional_calming": [
        "There is a calming effect that helps me destress",
        "Calming sensation that eases my mind",
    ],
    "emotional_comforting": [
        "Using this is so comforting, like a warm hug",
        "Provides comforting relief whenever I need it",
    ],
    "emotional_peaceful": [
        "It gives me a peaceful feeling after use",
        "Creates a peaceful atmosphere in my bedroom",
    ],
    "emotional_selfcare": [
        "A little self-care ritual that I look forward to",
        "Self-care product that actually makes a difference",
    ],
    "emotional_unwind": [
        "Helps me unwind after a stressful day",
        "Perfect way to unwind before bedtime",
    ],
}

FRICTION_SEEDS = {
    "Sensory_Overload": [
        "light too bright hurts eyes",
        "made me dizzy",
        "glare too intense",
        "need goggles to use",
        "eye strain from light",
    ],
    "Insufficient_Heat": [
        "does not heat to the set temperature",
        "not hot enough to feel any effect",
        "takes too long to warm up",
        "power too low to make a difference",
    ],
    "EMF_Radiation_Concern": [
        "EMF detector went off despite claims of zero emissions",
        "advertised as no EMF but radiation was detected",
        "concerned about high electromagnetic radiation from this device",
    ],
    "Stand_Adjustment_Issue": [
        "stand is stiff and hard to adjust",
        "difficult to move the arm",
        "cannot adjust panel angle",
    ],
    "Button_Sensitivity": [
        "buttons too sensitive accidentally hit",
        "timer button easily triggered",
        "accidental presses on control",
    ],
    "Control_Placement_Issue": [
        "switch should be on the cord",
        "hard to see the settings",
        "controls are awkward to reach",
    ],
    "Cord_Adapter_Issue": [
        "power cord too short to reach",
        "bulky adapter blocks outlet",
        "short cord makes placement difficult",
    ],
    "LED_Density_Insufficient": [
        "LED density lower than smaller model",
        "same number of LEDs on larger mat",
        "light output too weak for size",
    ],
    "Adverse_Reaction": [
        "chemical burns on face",
        "severe blistering and swelling",
        "terrible rash needed medication",
        "permanent scarring and hyperpigmentation",
        "blood pressure dropped after use",
        "felt lightheaded and dizzy",
        "caused me to nearly faint",
        "was left with dark spots",
        "hyperpigmentation after use",
    ],
    "Skin_Compatibility": [
        "not for melanated skin",
        "dark spots on brown skin",
        "black dots after use",
        "caution black women against using",
    ],
    "Build_Quality": [
        "fell apart during first use",
        "plastic cracked after minimal handling",
        "poorly manufactured broke within days",
    ],
    "Wearing_Discomfort": [
        "very uncomfortable to wear",
        "cannot find comfortable position",
        "awkward and unbearable",
    ],
    "Breath_Moisture": [
        "foggy and damp from breathing",
        "moisture inside mask",
        "fogs up like covid mask",
    ],
    "Coverage_Insufficient": [
        "does not cover eleven lines",
        "misses crow feet areas",
        "does not treat forehead wrinkles",
    ],
    "Thermal_Discomfort": [
        "panel gets too hot",
        "overheated after few minutes",
        "uncomfortable warmth like heater",
    ],
    "Posture_Burden": [
        "neck gets tired standing",
        "hard to maintain posture",
        "back hurts from position",
    ],
    "Fit_Issue": [
        "mask does not fit properly",
        "straps too long for small head",
        "keeps sliding down nose",
        "neck massager too big",
        "elbow brace too tight",
        "strap too loose falls off",
    ],
    "Eyewear_Incompatibility": [
        "cannot wear glasses with mask",
        "glasses not fit underneath",
        "cannot use reading glasses",
    ],
    "Device_Handling": [
        "device heavy hard to move",
        "bulky difficult to set up",
        "carrying is a struggle",
    ],
    "Activation_Energy": [
        "setup is total hassle",
        "force myself to turn on",
        "inconvenient to set up skip often",
    ],
    "Consistency_Burden": [
        "used daily then forgot",
        "hard to keep up routine",
        "cannot stay consistent",
    ],
    "Time_Burden": [
        "takes too many minutes each session",
        "no time to use as recommended",
        "time commitment too high",
    ],
    "Routine_Disruption": [
        "cannot fit into daily schedule",
        "disrupts normal routine",
        "does not fit lifestyle",
    ],
    "Protocol_Confusion": [
        "not sure how far to stand",
        "unclear how long to use",
        "user manual confusing",
    ],
    "Efficacy_Ambiguity": [
        "no improvement after months",
        "did nothing for my skin",
        "no results total waste",
    ],
    "Information_Overload": [
        "too many settings overwhelmed",
        "do not understand wavelengths",
        "technical details confusing",
    ],
    "Transparency_Concern": [
        "product description misleading",
        "changed technology without notice",
        "not enough information to trust",
    ],
    "Decision_Fatigue": [
        "do not know which mode",
        "too many choices",
        "options paralyzing wish simpler",
    ],
    "Feels_Like_Chore": [
        "feels like another chore",
        "force myself to use not enjoyable",
        "task not self care",
    ],
    "Guilt_Loop": [
        "should use more but put off",
        "collecting dust feel guilty",
        "spent money never use",
    ],
    "Disappointment": [
        "expected more did not live up",
        "disappointed with results",
        "not what i thought underwhelming",
    ],
    "Anxiety": [
        "worried about eye damage",
        "not sure if safe daily",
        "concerns about long term effects",
    ],
    "Space_Occupancy": [
        "takes up too much room",
        "cannot find storage place",
        "footprint too large for desk",
    ],
    "Aesthetic_Conflict": [
        "looks like medical equipment",
        "spouse hates how it looks",
        "ugly eyesore design",
    ],
    "Household_Conflict": [
        "wife hates bright light",
        "partner complains about noise",
        "bothers kids in common area",
    ],
    "Shipping_Damage": [
        "arrived broken unusable",
        "box crushed item shattered",
        "cracked screen return",
    ],
    "Regulatory_Approval_Issue": [
        "not FDA approved",
        "not eligible for FSA",
        "cannot use HSA funds",
        "missing medical certification",
    ],
    "Missing_Parts": [
        "missing parts to hang it",
        "no hardware included",
        "arrived incomplete",
    ],
    "Setup_Complexity": [
        "assembly took forever",
        "nightmare to put together",
        "needed help not user friendly",
    ],
    "Registration_Hassle": [
        "registering impossible website crash",
        "registration nightmare want data",
        "could not register device",
    ],
    "Reliability_Issue": [
        "died after two weeks",
        "malfunction right out of box",
        "broke within days",
    ],
    "Auto_Shutoff_Disruption": [
        "4 hour auto shutoff wakes me up",
        "timer too short disrupts sleep",
        "cannot disable auto power off",
    ],
    "Material_Degradation": [
        "material disintegrated after use",
        "fabric shed particles and became messy",
        "cover peeled off over time",
    ],
    "Battery_Life": [
        "battery dies after hour",
        "does not hold charge",
        "battery runs out quickly",
    ],
    "Charging_Issue": [
        "not charging blinking error",
        "stops charging shows error code",
        "charging problems out of box",
    ],
    "Customer_Support": [
        "customer service never replied",
        "warranty process painful",
        "support was unhelpful",
    ],
    "Spam_Harassment": [
        "spam texts cannot stop",
        "different number each time",
        "bombarded with unwanted messages",
    ],
    "Warranty_Deception": [
        "scanned code got spam warranty fake",
        "qr code messed up phone",
        "qr code asked for credit card",
        "do not use qr codes scam",
    ],
    "Return_Friction": [
        "return shipping expensive kept it",
        "return process complicated gave up",
        "wanted to return too much hassle",
        "not refundable stuck with it",
    ],
    "Price_Shock": [
        "too expensive for what it is",
        "almost did not buy high price",
        "cost prohibitive hesitated",
    ],
    "Value_Doubt": [
        "complete waste of money",
        "overpriced for what you get",
        "feel cheated quality awful",
    ],
    "ROI_Uncertainty": [
        "not convinced good use of money",
        "expensive gamble wish not spent",
        "cannot decide worth price may return",
    ],
    "Medical_Device_Identity": [
        "room feels like hospital clinic",
        "design too clinical not aesthetic",
        "looks like doctors office",
    ],
    "Category_Confusion": [
        "thought it was a heater",
        "not sure what device is looks odd",
        "friends ask if medical equipment",
    ],
}

MOTIVATION_SEEDS = {
    "Back_Pain": [
        "bought for lower back pain",
        "back was killing me ordered",
        "purchased for back pain relief",
    ],
    "Joint_Pain": [
        "got for knee arthritis",
        "joint pain in hands",
        "shoulder discomfort led to buy",
    ],
    "Chronic_Pain": [
        "living with chronic pain",
        "constant pain drove purchase",
        "needed ongoing pain management",
    ],
    "Anti_Aging": [
        "reduce fine lines and wrinkles",
        "anti aging benefits reason",
        "looking younger important",
    ],
    "Skin_Health": [
        "clear acne improve skin",
        "fix skin condition",
        "needed something for skin problems",
    ],
    "Muscle_Recovery": [
        "needed muscle recovery after workouts",
        "sore muscles after gym",
        "speed up post exercise recovery",
    ],
    "Surgery_Recovery": [
        "bought to help with surgery recovery",
        "purchased for post-surgery healing",
        "using after my breast reconstruction surgery",
    ],
    "Exercise_Recovery": [
        "running recovery main reason",
        "faster recovery after cycling",
        "sports recovery important",
    ],
    "Sleep_Improvement": [
        "help with insomnia",
        "better sleep motivation",
        "poor sleep quality ordered",
    ],
    "Stress_Reduction": [
        "relax after stressful days",
        "stress relief main reason",
        "unwind and calm mind",
    ],
    "Personal_Recommendation": [
        "friend recommended this",
        "chiropractor suggested for back",
        "physical therapist told me buy",
        "trainer recommended for recovery",
        "wife insisted try this",
        "family member told me",
    ],
    "Professional_Recommendation": [
        "My doctor recommended this device.",
        "The dermatologist suggested I use this.",
        "Prescribed by my physician.",
        "My physical therapist advised this treatment.",
        "A specialist told me to buy this product.",
        "Following my clinician's recommendation.",
    ],
    "Online_Reviews": [
        "positive amazon reviews convinced",
        "read good things had to try",
        "high ratings and reviews trusted",
    ],
    "Influencer_Media": [
        "saw on youtube purchased",
        "tiktok video made me buy",
        "influencer recommended got one",
    ],
    "Discount_Used": [
        "I bought this because it was on sale and I couldn't pass up the deal.",
        "Had a coupon code, so I decided to finally get it.",
        "The Lightning Deal pushed me to purchase immediately.",
    ],
    "Gift_Purchase": [
        "birthday present for mom",
        "christmas gift for husband",
        "gift idea for friend",
    ],
    "Replacement_Needed": [
        "old one broke needed replacement",
        "upgraded from previous model",
        "replacing worn out device",
    ],
    "Energy_Boost": [
        "bought for low energy",
        "needed an energy boost",
        "more energy during the day",
    ],
    "General_Pain_Relief": [
        "bought for pain relief",
        "needed something for aches and pains",
        "purchased to relieve daily pain",
    ],
    "Mood_Improvement": [
        "bought to improve mood",
        "needed help with depression",
        "purchased to lift my spirits",
    ],
}

TIME_PATTERNS = {
    "Morning": [
        r"\b(?:in\s+the\s+morning|morning|before\s+breakfast|after\s+waking\s+up|early\s+morning|start\s+of\s+the\s+day)\b"
    ],
    "Afternoon": [
        r"\b(?:in\s+the\s+afternoon|afternoon|midday|noon|lunch\s+break|after\s+lunch|early\s+afternoon)\b"
    ],
    "Evening": [
        r"\b(?:in\s+the\s+evening|evening|after\s+work|after\s+office|before\s+dinner|dinner\s+time)\b"
    ],
    "Night": [r"\b(?:at\s+night|night|late\s+night|overnight|during\s+the\s+night)\b"],
    "Before_Bed": [
        r"\b(?:before\s+bed|before\s+sleep|bedtime|going\s+to\s+bed|fall\s+asleep|wind\s+down\s+(?:for|before)\s+(?:bed|sleep))\b"
    ],
}

LOCATION_PATTERNS = {
    "Home": [
        r"\b(?:at\s+home(?:s)?|home\s+use|in\s+my\s+(?:house(?:s)?|apartment(?:s)?|room(?:s)?|bedroom(?:s)?|bathroom(?:s)?|living\s+room(?:s)?|family\s+room(?:s)?))\b",
        r"\b(?:around\s+the\s+house(?:s)?|for\s+home\s+use)\b",
    ],
    "Office": [
        r"\b(?:at\s+(?:the\s+)?office|in\s+my\s+(?:home\s+)?office|workplace|while\s+working|at\s+my\s+desk)\b",
        r"\b(?:for|to|into)\s+my\s+office\b",
    ],
    "Gym": [
        r"\b(?:at\s+the\s+gym(?:s)?|gym\s+session(?:s)?|fitness\s+center(?:s)?|while\s+working\s+out|at\s+the\s+studio(?:s)?)\b"
    ],
    "Clinic": [
        r"\b(?:at\s+(?:the|a|my)\s+(?:clinic(?:s)?|doctor'?s?\s+office(?:s)?|medical\s+office(?:s)?|physical\s+therapy\s+clinic(?:s)?|rehab\s+center(?:s)?|hospital(?:s)?|chiropractor'?s?\s+office(?:s)?))\b"
    ],
    "Spa": [
        r"\b(?:at\s+(?:the|a|my)\s+(?:spa(?:s)?|salon(?:s)?|beauty\s+salon(?:s)?|hair\s+salon(?:s)?|medspa(?:s)?|aesthetic\s+clinic(?:s)?|wellness\s+center(?:s)?))\b"
    ],
    "Travel": [
        r"\b(?:while\s+travel(?:ing|led)?|on\s+(?:the|a)\s+(?:plane(?:s)?|train(?:s)?|road|trip(?:s)?|vacation(?:s)?)|in\s+my\s+hotel(?:s)?|portable\s+for\s+travel|on\s+the\s+go)\b"
    ],
}

# ===================== 编译正则和预计算种子向量 =====================
compiled_subs = {
    k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in SUB_PATTERNS.items()
}
compiled_times = {
    k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in TIME_PATTERNS.items()
}
compiled_locations = {
    k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in LOCATION_PATTERNS.items()
}

compiled_friction_re = {}
friction_sub_to_main = {}
for main, subs in FRICTION_TAXONOMY.items():
    for sub, patterns in subs.items():
        compiled_friction_re[sub] = [re.compile(p, re.IGNORECASE) for p in patterns]
        friction_sub_to_main[sub] = main

compiled_motivation_re = {}
motivation_sub_to_main = {}
for main, subs in MOTIVATION_TAXONOMY.items():
    for sub, patterns in subs.items():
        compiled_motivation_re[sub] = [re.compile(p, re.IGNORECASE) for p in patterns]
        motivation_sub_to_main[sub] = main

# 场景种子向量
scene_vectors = {}
for scene, seeds in SCENE_SEEDS.items():
    emb = embed_model.encode(seeds, convert_to_tensor=True).mean(dim=0)
    scene_vectors[scene] = emb

# 摩擦种子向量
friction_seed_vectors = {}
for sub, seeds in FRICTION_SEEDS.items():
    emb = embed_model.encode(seeds, convert_to_tensor=True).mean(dim=0)
    friction_seed_vectors[sub] = emb

# 动机种子向量
motivation_seed_vectors = {}
for sub, seeds in MOTIVATION_SEEDS.items():
    emb = embed_model.encode(seeds, convert_to_tensor=True).mean(dim=0)
    motivation_seed_vectors[sub] = emb

MOTIVATION_NO_SEMANTIC_FALLBACK = {
    "Professional_Recommendation",
    "Personal_Recommendation",
    "Online_Reviews",
    "Influencer_Media",
}


def encode_reviews(texts, batch_size=128):
    """批量编码评论文本，返回嵌入 tensor (N, 384)"""
    return embed_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )


def empty_feature():
    """返回空的桥表结构及空 Keywords"""
    return {
        "Scenes": [],
        "Frictions": [],
        "Motivations": [],
        "Times": [],
        "Locations": [],
        "Keywords": "",
    }


def generate_keyword_text(text):
    """生成分号分隔的词干序列（保留频率，不去重），用于词云"""
    if not text:
        return ""
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        lemma = token.lemma_.lower()
        if len(lemma) >= 3 or lemma in DOMAIN_KEYWORDS:
            tokens.append(lemma)
    return ";".join(tokens)


def classify_review(clean_text, rating, emb=None):
    if clean_text is None:
        return empty_feature()
    clean_text = str(clean_text).strip()
    if not clean_text:
        return empty_feature()

    if emb is None:
        emb = embed_model.encode([clean_text], convert_to_tensor=True)

    # ---- 场景识别 ----
    matched_subs = []
    for sub_key, patterns in compiled_subs.items():
        for pat in patterns:
            if pat.search(clean_text):
                matched_subs.append(sub_key)
                break
    seen = set()
    unique_subs = sorted([s for s in matched_subs if not (s in seen or seen.add(s))])

    scene_list = []
    if unique_subs:
        for sub in unique_subs:
            main = MAIN_SCENE_MAP[sub]
            scene_list.append({"Main": main, "Sub": sub})
    else:
        sims = {sc: util.cos_sim(emb, vec).item() for sc, vec in scene_vectors.items()}
        best_scene = max(sims, key=sims.get)
        best_score = sims[best_scene]
        if best_score >= SEMANTIC_THRESHOLD:
            sorted_scores = sorted(sims.values(), reverse=True)
            margin = (
                sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
            )
            if margin >= SCENE_MARGIN:
                main = MAIN_SCENE_MAP[best_scene]
                scene_list.append({"Main": main, "Sub": best_scene})

    # ---- 时间识别 ----
    time_slots = []
    for time_label, patterns in compiled_times.items():
        for pat in patterns:
            if pat.search(clean_text):
                time_slots.append(time_label)
                break
    time_list = list(dict.fromkeys(time_slots))

    # ---- 地点识别 ----
    location_slots = []
    for loc_label, patterns in compiled_locations.items():
        for pat in patterns:
            if pat.search(clean_text):
                location_slots.append(loc_label)
                break
    location_list = list(dict.fromkeys(location_slots))

    # ---- 摩擦识别（完整后处理） ----
    friction_subs = set()
    friction_source = {}
    friction_score = {}
    friction_margin = {}

    for sub, patterns in compiled_friction_re.items():
        for pat in patterns:
            if pat.search(clean_text):
                friction_subs.add(sub)
                friction_source[sub] = "Regex"
                friction_score[sub] = None
                friction_margin[sub] = None
                break

    # 后处理（你的完整逻辑）
    if "Reliability_Issue" in friction_subs:
        if re.search(
            r"\b(?:another|old|previous|other)\s+(?:style|one|mask|device|panel|light|model|company)\b.*?\bstopped\s+working\b",
            clean_text,
            re.I,
        ) or re.search(
            r"\ban?\s+\w+\s+I\s+used\s+for\s+years\s+stopped\s+working\b",
            clean_text,
            re.I,
        ):
            friction_subs.discard("Reliability_Issue")
            friction_source.pop("Reliability_Issue", None)
            friction_score.pop("Reliability_Issue", None)
            friction_margin.pop("Reliability_Issue", None)
    if "Build_Quality" in friction_subs and re.search(
        r"\b(?:excellent\s+quality|solid\s+construction|great\s+design|well\s+made|works\s+great)\b",
        clean_text,
        re.I,
    ):
        friction_subs.discard("Build_Quality")
        friction_source.pop("Build_Quality", None)
        friction_score.pop("Build_Quality", None)
        friction_margin.pop("Build_Quality", None)
    if "Device_Handling" in friction_subs and re.search(
        r"\b(?:(?:not|never)\s+(?:\w+\s+){0,2}heavy|(?:won't|will\s+not|wouldn't|would\s+not|doesn't|does\s+not|don't|do\s+not|isn't|is\s+not)\s+(?:\w+\s+){0,2}heavy)\b",
        clean_text,
        re.I,
    ):
        friction_subs.discard("Device_Handling")
        friction_source.pop("Device_Handling", None)
        friction_score.pop("Device_Handling", None)
        friction_margin.pop("Device_Handling", None)
    if (
        "Efficacy_Ambiguity" in friction_subs
        and re.search(r"\bdid\s+not\s+work\b", clean_text, re.I)
        and re.search(
            r"\b(?:broken|defective|malfunction|stopped|broke|faulty|died|dead)\b",
            clean_text,
            re.I,
        )
    ):
        friction_subs.discard("Efficacy_Ambiguity")
        friction_source.pop("Efficacy_Ambiguity", None)
        friction_score.pop("Efficacy_Ambiguity", None)
        friction_margin.pop("Efficacy_Ambiguity", None)
    if "Posture_Burden" in friction_subs and not re.search(
        r"\b(?:neck\s+(?:pain|sore|hurt|tired)\s+(?:from|when|while|using|caused\s+by|because\s+of)|(?:makes?|causes?|gives?\s+me)\s+(?:my\s+)?neck\s+(?:pain|sore|hurt|tired))",
        clean_text,
        re.I,
    ):
        friction_subs.discard("Posture_Burden")
        friction_source.pop("Posture_Burden", None)
        friction_score.pop("Posture_Burden", None)
        friction_margin.pop("Posture_Burden", None)
    if "Battery_Life" in friction_subs and re.search(
        r"\b(?:battery\s+life\s+(?:has\s+been\s+)?(?:solid|great|good|excellent|amazing|impressive|fine|decent|acceptable|ok|okay)|(?:amazing|great|good|excellent|solid|impressive|fantastic|decent|awesome|superb)\s+battery\s+life|(?:no|zero)\s+(?:issues?|problems?|complaints?)\s+with\s+(?:the\s+)?battery)\b",
        clean_text,
        re.I,
    ):
        friction_subs.discard("Battery_Life")
        friction_source.pop("Battery_Life", None)
        friction_score.pop("Battery_Life", None)
        friction_margin.pop("Battery_Life", None)
    if (
        "Price_Shock" in friction_subs
        and rating is not None
        and rating >= 4
        and re.search(
            r"\b(?:good\s+value|on\s+a\s+budget|worth\s+(?:it|the\s+money)|great\s+price|affordable|for\s+the\s+price|half\s+the\s+price|best\s+value|reasonable\s+price|well\s+priced|unbeatable\s+price|excellent\s+price|fair\s+price|good\s+deal|bargain)\b",
            clean_text,
            re.I,
        )
    ):
        friction_subs.discard("Price_Shock")
        friction_source.pop("Price_Shock", None)
        friction_score.pop("Price_Shock", None)
        friction_margin.pop("Price_Shock", None)
    if "Thermal_Discomfort" in friction_subs and re.search(
        r"\b(?:doesn'?t\s+feel\s+too\s+hot|not\s+hot|comfortable\s+temperature|doesn'?t\s+get\s+hot)\b",
        clean_text,
        re.I,
    ):
        friction_subs.discard("Thermal_Discomfort")
        friction_source.pop("Thermal_Discomfort", None)
        friction_score.pop("Thermal_Discomfort", None)
        friction_margin.pop("Thermal_Discomfort", None)

    if not friction_subs:
        scores = {
            sub: util.cos_sim(emb, vec).item()
            for sub, vec in friction_seed_vectors.items()
        }
        best = max(scores, key=scores.get)
        best_score = scores[best]
        if best_score >= FRICTION_THRESHOLD:
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = (
                sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
            )
            if margin >= FRICTION_MARGIN and not (
                rating is not None
                and rating == 5
                and best in ("Adverse_Reaction", "Efficacy_Ambiguity")
            ):
                friction_subs.add(best)
                friction_source[best] = "Embedding"
                friction_score[best] = round(best_score, 3)
                friction_margin[best] = round(margin, 3)

    if (
        "Disappointment" in friction_subs
        and rating is not None
        and rating == 5
        and re.search(
            r"\b(?:worth\s+the\s+money|very\s+comfortable|not\s+bulky|really\s+great|so\s+comfortable|well\s+worth\s+it|love\s+(?:it|this|mine)|highly\s+recommend)\b",
            clean_text,
            re.I,
        )
    ):
        friction_subs.discard("Disappointment")
        friction_source.pop("Disappointment", None)
        friction_score.pop("Disappointment", None)
        friction_margin.pop("Disappointment", None)

    friction_list = []
    for sub in sorted(friction_subs):
        friction_list.append(
            {
                "Main": friction_sub_to_main[sub],
                "Sub": sub,
                "Source": friction_source.get(sub, ""),
                "Score": friction_score.get(sub),
                "Margin": friction_margin.get(sub),
            }
        )

    # ---- 动机识别 ----
    motivation_subs = set()
    motivation_source = {}
    motivation_score = {}
    motivation_margin = {}

    for sub, patterns in compiled_motivation_re.items():
        for pat in patterns:
            if pat.search(clean_text):
                motivation_subs.add(sub)
                motivation_source[sub] = "Regex"
                motivation_score[sub] = None
                motivation_margin[sub] = None
                break

    if not motivation_subs:
        scores = {
            sub: util.cos_sim(emb, vec).item()
            for sub, vec in motivation_seed_vectors.items()
            if sub not in MOTIVATION_NO_SEMANTIC_FALLBACK
        }
        if scores:
            best = max(scores, key=scores.get)
            best_score = scores[best]
            if best_score >= MOTIVATION_THRESHOLD:
                sorted_scores = sorted(scores.values(), reverse=True)
                margin = (
                    sorted_scores[0] - sorted_scores[1]
                    if len(sorted_scores) > 1
                    else 1.0
                )
                if margin >= MOTIVATION_MARGIN:
                    motivation_subs.add(best)
                    motivation_source[best] = "Embedding"
                    motivation_score[best] = round(best_score, 3)
                    motivation_margin[best] = round(margin, 3)

    motivation_list = []
    for sub in sorted(motivation_subs):
        motivation_list.append(
            {
                "Main": motivation_sub_to_main[sub],
                "Sub": sub,
                "Source": motivation_source.get(sub, ""),
                "Score": motivation_score.get(sub),
                "Margin": motivation_margin.get(sub),
            }
        )

    # ---- 生成 Keywords ----
    keywords = generate_keyword_text(clean_text)

    return {
        "Scenes": scene_list,
        "Frictions": friction_list,
        "Motivations": motivation_list,
        "Times": time_list,
        "Locations": location_list,
        "Keywords": keywords,
    }
