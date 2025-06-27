import nltk
from nltk import CFG
import re

# --- Preprocessing Function ---
def clean_question(text):
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")  # Normalize apostrophes
    text = re.sub(r"[–—]", "-", text)  # Normalize dashes
    text = re.sub(r"[^\w\s']", '', text)  # Remove all punctuation except apostrophes

    # Fix glued words like "applicationhow", "showingwhat"
    stuck_words = {
        r'(application)(how)': r'\1 \2',
        r'(application)(can)': r'\1 \2',
        r'(application)(do)': r'\1 \2',
        r'(showing)(what)': r'\1 \2',
        r'(deadline)(can)': r'\1 \2',
        r'(mistake)(on)': r'\1 \2',
        r'(missed)(the)': r'\1 \2',
    }
    for pattern, replacement in stuck_words.items():
        text = re.sub(pattern, replacement, text)

    # Contraction expansions
    contractions = {
        "what's": "what is",
        "i've": "i have",
        "isn't": "is not",
        "can't": "cannot",
        "don't": "do not",
        "doesn't": "does not",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is"
    }
    for c, full in contractions.items():
        text = text.replace(c, full)

    # Common OCR or speech spelling errors
    corrections = {
        "infmation": "information",
        "ganization": "organization",
        "advis": "advisor",
        "brow": "borrow",
        "recds": "records",
        "certiforicate": "certificate",
        "reforund": "refund",
        "ofor": "of",
        "forrom": "from",
        "foree": "fee",
        "wkstudy": "workstudy",
        "forinal": "final",
        "deforer": "defer",
        "deforerment": "deferment",
        "transforer": "transfer",
        "forix": "fix",
        "f": "for",
        "lost and found": "lost and found on campus",  
    }

    words = text.split()
    words = [corrections.get(word, word) for word in words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spacing
    return text


# --- Grammar ---
grammar = CFG.fromstring("""
  S -> WH AUX NP VP
  S -> WH AUX VP
  S -> WH NP VP
  S -> AUX NP VP
  S -> NP VP
  S -> WH VP
  S -> VP

  WH -> 'how' | 'what' | 'where' | 'when' | 'who' | 'can' | 'is' | 'are'| 'ε'
  AUX -> 'do' | 'can' | 'will' | 'is' | 'are' | 'have'| 'there'| 'i' |should | 'ε'
  NP -> 'i' | 'we' | 'students' | 'alumni' | 'my'| 'a'| 'the' |'get'| 'an'  |'ε'

  VP -> 'join' 'a' 'student' 'club'
  VP -> 'join' 'a' 'student' 'organization'
  VP -> 'join' 'a' 'student' 'club' 'or''organization'
  VP -> 'speak' 'to' 'about' 'academic' 'counseling'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'dean'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'advisor'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'dean' 'advisor'
  VP -> 'replace' 'a' 'lost' 'student' 'id'
  VP -> 'collect' 'my' 'student' 'id'
  VP -> 'a' 'certificate' 'of' 'enrollment'
  VP -> 'a' 'student' 'status' 'letter'
  VP -> 'apply' 'for' 'graduation'
  VP ->  'graduation' 'requirements'
  VP -> 'graduation' 'ceremony' 'take' 'place'
  VP -> 'get' 'my' 'degree' 'certificate'
  VP -> 'what' 'should' 'i' 'do' 'if' 'my' 'name' 'is' 'misspelled' 'on' 'my' 'certificate'
  VP -> 'are' 'transcripts' 'sent' 'to' 'employers'
  VP -> 'are' 'transcripts' 'sent' 'directly' 
  VP -> 'are' 'transcripts' 'sent' 'to' 'other' 'schools'
  VP -> 'verify' 'my' 'academic' 'records'
  VP -> 'get' 'help' 'with' 'internships'
  VP -> 'get' 'help' 'with' 'job' 'placement'
  VP -> 'request' 'services' 'from' 'the' 'office'
  VP -> 'access' 'past' 'exam' 'papers'
  VP -> 'access' 'academic' 'resources'
  VP -> 'pay' 'in' 'installments'
  VP -> 'apply' 'for' 'a' 'scholarship'
  VP -> 'apply' 'for' 'a' 'bursary'
  VP -> 'get' 'financial' 'aid'
  VP -> 'i' 'have' 'paid' 'but' 'my' 'payment' 'is' 'not' 'showing'
  VP -> 'get' 'a' 'fee' 'structure'
  VP -> 'get' 'an' 'invoice'
  VP -> 'get' 'a' 'fee' 'structure' 'invoice'
  VP -> 'are' 'there' 'penalties' 'for' 'late' 'fee' 'payment'
  VP -> 'get' 'a' 'refund' 'if' 'i' 'withdraw'
  VP -> 'apply' 'for' 'work' 'study' 'program'
  VP -> 'are' 'there' 'work' 'study' 'programs'
  VP -> 'apply' 'for' 'a' 'student' 'loan'
  VP -> 'register' 'for' 'my' 'courses'
  VP -> 'drop' 'a' 'course'
  VP -> 'add' 'a' 'course'
  VP -> 'get' 'my' 'class' 'schedule'
  VP -> 'access' 'my' 'academic' 'transcript'
  VP -> 'get' 'a' 'letter' 'of' 'recommendation'
  VP -> 'get' 'a' 'letter' 'of' 'recommendation' 'from' 'the' 'registrar'
  VP -> 'what' 'is' 'the' 'grading' 'system' 'here'
  VP -> 'miss' 'an' 'exam'
  VP -> 'apply' 'for' 'academic' 'leave'
  VP -> 'defer' 'my' 'admission'
  VP -> 'apply' 'to' 'this' 'university'
  VP -> 'what' 'documents' 'do' 'i' 'need' 'for' 'my' 'application'
  VP -> 'track' 'my' 'admission' 'status'
  VP -> 'track' 'my' 'admission' 'status' 'online'
  VP -> 'i' 'missed' 'the' 'deadline'
  VP -> 'can' 'i' 'still' 'apply'
  VP -> 'what' 'are' 'the' 'minimum' 'entry' 'requirements'
  VP -> 'transfer' 'credits' 'from' 'another' 'institution'
  VP -> 'option'
  VP -> 'apply' 'for' 'a' 'second' 'degree'
  VP -> 'i' 'made' 'a' 'mistake' 'on' 'my' 'application'
  VP -> 'how' 'can' 'i' 'fix' 'it'
  VP -> 'how' 'can' 'i' 'fix' 'my' 'application'
  VP -> 'receive' 'my' 'admission' 'letter'
  VP -> 'find' 'housing' 'information'
  VP -> 'when' 'will' 'my' 'final' 'results' 'be' 'released'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'dean'
  VP -> 'book' 'an' 'appointment' 'with' 'my' 'advisor'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'dean' 'or' 'advisor'
  VP -> 'lost' 'and' 'found' 'on''campus'
  VP -> 'borrow' 'books'
  VP -> 'where' 'is' 'the' 'library'
  VP -> 'borrow' 'books' 'from' 'the' 'library'
  VP -> 'health' 'center' 'on' 'campus'
  VP -> 'get' 'a' 'certificate' 'of' 'enrollment' 'or' 'student' 'status' 'letter'
  VP -> 'apply' 'to' 'be' 'a' 'student' 'representative'
  VP -> 'minimum' 'entry' 'requirements'
  VP -> 'transfer' 'credits' 'from' 'another' 'institution'
  VP -> 'access' 'past' 'exam' 'papers'
  VP -> 'access' 'academic' 'resources'
  VP -> 'access' 'past' 'exam' 'papers' 'academic' 'resources'
  VP -> 'pay' 'my' 'tuition' 'fees'
  VP -> 'pay' 'tuition' 'fees'
  VP -> 'apply' 'for' 'a' 'scholarship'
  VP -> 'apply' 'for' 'a' 'bursary'
  VP -> 'apply' 'for' 'a' 'scholarship' 'bursary'
  VP -> 'what' 'financial' 'aid' 'options' 'are' 'available'
  VP -> 'i' 'have' 'paid' 'but' 'my' 'payment' 'is' 'not' 'showing'
  VP -> 'get' 'a' 'fee' 'structure'
  VP -> 'get' 'an' 'invoice'
  VP -> 'get' 'a' 'fee' 'structure' 'invoice'
  VP -> 'what' 'is' 'the' 'procedure' 'for' 'dropping' 'a' 'course'
  VP -> 'what' 'is' 'the' 'procedure' 'for' 'adding' 'a' 'course'
  VP -> 'what' 'is' 'the' 'procedure' 'for' 'dropping' 'or' 'adding' 'a' 'course'
  VP -> 'what' 'happens' 'if' 'i' 'miss' 'an' 'exam'


                         

                       


""")

# --- Category and Response Maps ---
category_map = {
    'join a student club or organization': 'Clubs & Activities',
    'join a student club': 'Clubs & Activities',
    'join a student organization': 'Clubs & Activities',
    'book an appointment with the dean or advisor': 'Appointments',
    'get a fee structure invoice': 'Finance & Fees',
    'what should i do if my name is misspelled on my certificate': 'Graduation & Certification',
    'transfer credits from another institution': 'Admissions',
    'are transcripts sent to other schools': 'Transcripts & Results',
    'are transcripts sent to employers': 'Transcripts & Results',
    'get help with internships': 'Career & Opportunities',
    'get help with job placement': 'Career & Opportunities',
    'request services from the office': 'Administration',
    'are there work study programs': 'Career & Opportunities',
    'what are the minimum entry requirements': 'Admissions',
    'transfer credits from another institution': 'Admissions',
    'minimum entry requirements': 'Admissions',
    'how can i fix my application': 'Admissions',
    'track my admission status online': 'Admissions',
    'lost and found on campus': 'Lost & Found / ID',
    'where is the library': 'Library & Academic Resources',
    'borrow books from the library': 'Library & Academic Resources',
    'health center on campus': 'Student Services',
    'get a certificate of enrollment or student status letter': 'Letters & Certificates',
    'apply to be a student representative': 'Student Leadership & Governance',
    'a certificate of enrollment':'Letters & Certificates',
    'get my degree certificate':'Letters & Certificates',
    'a student status letter':'Letters & Certificates',
    'graduation requirements': 'Graduation & Certification',
    'graduation ceremony take place': 'Graduation & Certification',
    'access past exam papers': 'Library & Academic Resources',
    'access academic resources': 'Library & Academic Resources',
    'verify my academic records': 'Library & Academic Resources',
    'access past exam papers academic resources': 'Library & Academic Resources',
    'pay my tuition fees': 'Finance & Fees',
    'pay tuition fees': 'Finance & Fees',
    'pay in installments': 'Finance & Fees',
    'apply for a scholarship': 'Finance & Fees',
    'apply for a student loan': 'Finance & Fees',
    'apply for a bursary': 'Finance & Fees',
    'apply for a scholarship bursary': 'Finance & Fees',
    'what financial aid options are available': 'Finance & Fees',
    'i have paid but my payment is not showing': 'Finance & Fees',
    'get a fee structure': 'Finance & Fees',
    'get an invoice': 'Finance & Fees',
    'get a refund if i withdraw': 'Finance & Fees',
    'get a fee structure invoice': 'Finance & Fees',
    'are there penalties for late fee payment': 'Finance & Fees',
    'find housing information': 'Admissions',  
    'speak to about academic counseling':'Academic Support',
    'replace a lost student id':'Lost & Found / ID',
    'apply for graduation': 'Graduation & Certification',
    'register for my courses': 'Registration & Exams',
    'what is the procedure for dropping a course': 'Registration & Exams',
    'what is the procedure for adding a course': 'Registration & Exams',
    'what is the procedure for dropping or adding a course': 'Registration & Exams',
    'get my class schedule': 'Registration & Exams',
    'access my academic transcript': 'Transcripts & Results',
    'get a letter of recommendation from the registrar': 'Letters & Certificates',
    'what is the grading system here': 'Grading Information',
    'collect my student id': 'Lost & Found / ID',
    'what happens if i miss an exam': 'Registration & Exams',
    'apply for academic leave': 'Registration & Exams',
    'when will my final results be released': 'Transcripts & Results',
    'what documents do i need for my application': 'Admissions',
    'defer my admission': 'Admissions',
    'receive my admission letter': 'Admissions',
}

response_map = {
    'Clubs & Activities': "You can join student clubs through the student affairs office or during orientation.",
    'Academic Support': "Speak to the academic advisor at your department or counseling office.",
    'Appointments': "Book appointments with the dean or advisor through the department or portal.",
    'Lost & Found / ID': "Visit campus security or registrar for ID issues or lost items.",
    'Letters & Certificates': "Request letters and certificates from the registrar or online portal.",
    'Graduation & Certification': "Graduation info is on the portal. For name issues, visit the registrar.",
    'Transcripts & Results': "Check the academic portal for transcripts and final results.",
    'Career & Opportunities': "Internship and job placement help is available at the career office.",
    'Administration': "Alumni can request services through the alumni office.",
    'Library & Academic Resources': "Get past papers and other academic  resources from the library or academic portal.",
    'Finance & Fees': "Fee payments,penalties , refunds, and invoices are available on the finance portal.",
    'Registration & Exams': "Register, add/drop courses, and view schedules on the student portal.",
    'Admissions': "Check status, submit documents, and fix errors online or at admissions.",
    'Student Services': "Yes, there is a health center on campus. Visit the university clinic for medical services.",
    'Student Leadership & Governance': "You can apply to be a student representative during the nomination period through the student affairs office.",
    'Grading Information': "Our grading system uses a standard A-F scale. Each grade corresponds to a grade point and contributes to your GPA. You can view full grading policies in the academic handbook or on the student portal.",



}

# --- Parsing Function ---
parser = nltk.ChartParser(grammar)

def parse_question(question):
    text = question.lower().replace('?', '')
    tokens = text.split()
    print("DEBUG: Tokens:", tokens)

    try:
        for tree in parser.parse(tokens):
            print("✅ Parsed tree:")
            tree.pretty_print()

            vp_phrase = ' '.join(tokens[-len(tree.productions()[-1].rhs()):])
            print("DEBUG: VP phrase:", vp_phrase)

            category = category_map.get(vp_phrase, "General Inquiry")
            response = response_map.get(category, "This query has been received and is being processed.")
            return {
                "parsed": True,
                "category": category,
                "response": response
            }
        print("❌ No match in grammar.")
        return {"parsed": False}
    except Exception as e:
        print("⚠️ Grammar error:", e)
        return {"parsed": False, "error": str(e)}


