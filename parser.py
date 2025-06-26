import nltk
from nltk import CFG

# Broad CFG grammar with more question variations and synonyms
grammar = CFG.fromstring("""
  S -> WH AUX NP VP
  WH -> 'how' | 'what' | 'where' | 'when' | 'who' | 'can'
  AUX -> 'do' | 'can' | 'will' | 'is' | 'are'
  NP -> 'i' | 'students' | 'alumni' | 'we'

  VP -> 'check' 'my' 'final' 'results'
  VP -> 'see' 'my' 'final' 'results'
  VP -> 'view' 'my' 'final' 'results'
  VP -> 'get' 'my' 'final' 'results'
  VP -> 'access' 'my' 'final' 'results'
  VP -> 'find' 'my' 'final' 'results'
  VP -> 'when' 'will' 'my' 'final' 'results' 'be' 'released'
  VP -> 'know' 'when' 'final' 'results' 'are' 'released'

  VP -> 'apply' 'for' 'graduation'
  VP -> 'get' 'a' 'certificate'
  VP -> 'what' 'if' 'my' 'name' 'is' 'misspelled' 'on' 'my' 'certificate'
  VP -> 'correct' 'my' 'certificate'
  VP -> 'change' 'my' 'name' 'on' 'certificate'
  VP -> 'fix' 'a' 'mistake' 'on' 'my' 'certificate'
  VP -> 'get' 'my' 'degree' 'certificate'
  VP -> 'collect' 'my' 'degree' 'certificate'
  VP -> 'receive' 'my' 'degree' 'certificate'
  VP -> 'claim' 'my' 'certificate'
  VP -> 'how' 'do' 'i' 'get' 'my' 'certificate'
  VP -> 'replace' 'a' 'lost' 'student' 'id'
  VP -> 'collect' 'my' 'student' 'id'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'dean'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'advisor'
  VP -> 'schedule' 'a' 'meeting' 'with' 'the' 'dean'
  VP -> 'schedule' 'a' 'meeting' 'with' 'the' 'advisor'
  VP -> 'arrange' 'a' 'meeting' 'with' 'the' 'dean'
  VP -> 'arrange' 'a' 'meeting' 'with' 'the' 'advisor'
  VP -> 'set' 'up' 'a' 'meeting' 'with' 'the' 'dean'
  VP -> 'set' 'up' 'a' 'meeting' 'with' 'the' 'advisor'
  VP -> 'book' 'an' 'appointment' 'with' 'the' 'advisor'
  VP -> 'join' 'a' 'student' 'club'
  VP -> 'join' 'a' 'student' 'organization'
  VP -> 'join' 'a' 'campus' 'club'
  VP -> 'join' 'a' 'campus' 'organization'
  VP -> 'sign' 'up' 'for' 'a' 'student' 'club'
  VP -> 'sign' 'up' 'for' 'a' 'student' 'organization'
  VP -> 'participate' 'in' 'student' 'clubs'
  VP -> 'participate' 'in' 'campus' 'activities'
  VP -> 'become' 'a' 'member' 'of' 'a' 'student' 'club'
  VP -> 'become' 'a' 'member' 'of' 'a' 'student' 'organization'
  VP -> 'join' 'a' 'student' 'organisation'
  VP -> 'join' 'a' 'club'
  VP -> 'join' 'a' 'organization'
  VP -> 'apply' 'to' 'be' 'a' 'student' 'representative'
  VP -> 'run' 'for' 'student' 'representative'
  VP -> 'nominate' 'myself' 'as' 'student' 'representative'
  VP -> 'get' 'elected' 'as' 'student' 'leader'
  VP -> 'join' 'student' 'government'
  VP -> 'speak' 'to' 'someone' 'about' 'academic' 'counseling'
  VP -> 'talk' 'to' 'someone' 'about' 'academic' 'counseling'
  VP -> 'get' 'academic' 'counseling'
  VP -> 'meet' 'an' 'academic' 'advisor'
  VP -> 'meet' 'with' 'an' 'academic' 'advisor'
  VP -> 'see' 'an' 'academic' 'counselor'
  VP -> 'get' 'help' 'with' 'academic' 'issues'
  VP -> 'access' 'my' 'transcript'
  VP -> 'send' 'transcripts' 'to' 'employers'
  VP -> 'send' 'transcripts' 'to' 'other' 'schools'
  VP -> 'are' 'transcripts' 'sent' 'to' 'institutions'
  VP -> 'verify' 'my' 'academic' 'records'
  VP -> 'confirm' 'my' 'academic' 'records'
  VP -> 'check' 'my' 'academic' 'records'
  VP -> 'are' 'my' 'records' 'accurate'
  VP -> 'how' 'can' 'i' 'verify' 'my' 'records'
  VP -> 'get' 'a' 'letter' 'of' 'recommendation'
  VP -> 'request' 'a' 'letter' 'of' 'recommendation'
  VP -> 'apply' 'for' 'a' 'letter' 'of' 'recommendation'
  VP -> 'ask' 'for' 'a' 'letter' 'of' 'recommendation'
  VP -> 'get' 'a' 'student' 'status' 'letter'
  VP -> 'get' 'a' 'letter' 'confirming' 'enrollment'
  VP -> 'request' 'a' 'student' 'status' 'letter'
  VP -> 'request' 'a' 'certificate' 'of' 'enrollment'
  VP -> 'apply' 'for' 'a' 'student' 'status' 'letter'
  VP -> 'get' 'a' 'certificate' 'of' 'enrollment'
  VP -> 'register' 'for' 'my' 'courses'
  VP -> 'drop' 'a' 'course'
  VP -> 'add' 'a' 'course'
  VP -> 'miss' 'an' 'exam'
  VP -> 'access' 'past' 'exam' 'papers'
  VP -> 'access' 'academic' 'resources'
  VP -> 'get' 'past' 'papers'
  VP -> 'get' 'lecture' 'notes'
  VP -> 'download' 'study' 'materials'
  VP -> 'where' 'can' 'i' 'access' 'academic' 'resources'
  VP -> 'get' 'a' 'fee' 'structure'
  VP -> 'pay' 'my' 'tuition' 'fees'
  VP -> 'pay' 'in' 'installments'
  VP -> 'apply' 'for' 'a' 'scholarship'
  VP -> 'apply' 'for' 'a' 'bursary'
  VP -> 'get' 'financial' 'aid'
  VP -> 'request' 'a' 'refund'
  VP -> 'apply' 'for' 'a' 'student' 'loan'
  VP -> 'get' 'an' 'invoice'
  VP -> 'check' 'if' 'payment' 'was' 'received'
  VP -> 'where' 'can' 'i' 'pay' 'tuition' 'fees'
  VP -> 'how' 'do' 'i' 'pay' 'my' 'fees'
  VP -> 'get' 'a' 'fee' 'invoice'
  VP -> 'how' 'do' 'i' 'get' 'my' 'invoice'
  VP -> 'can' 'i' 'pay' 'fees' 'in' 'installments'
  VP -> 'what' 'financial' 'aid' 'options' 'are' 'available'
  VP -> 'i' 'paid' 'but' 'my' 'payment' "isn\'t" 'showing'
  VP -> 'are' 'there' 'penalties' 'for' 'late' 'fee' 'payment'
  VP -> 'can' 'i' 'get' 'a' 'refund' 'if' 'i' 'withdraw'
  VP -> 'face' 'penalties' 'for' 'late' 'fee' 'payment'
  VP -> 'apply' 'for' 'academic' 'leave'
  VP -> 'defer' 'my' 'admission'
  VP -> 'apply' 'to' 'this' 'university'
  VP -> 'submit' 'my' 'application' 'documents'
  VP -> 'track' 'my' 'admission' 'status'
  VP -> 'fix' 'a' 'mistake' 'on' 'my' 'application'
  VP -> 'receive' 'my' 'admission' 'letter'
  VP -> 'check' 'graduation' 'requirements'
  VP -> 'know' 'graduation' 'requirements'
  VP -> 'view' 'graduation' 'requirements'
  VP -> 'see' 'graduation' 'requirements'
  VP -> 'what' 'are' 'the' 'graduation' 'requirements'
  VP -> 'check' 'when' 'graduation' 'ceremony' 'takes' 'place'
  VP -> 'when' 'is' 'the' 'graduation' 'ceremony'
  VP -> 'when' 'will' 'graduation' 'happen'
  VP -> 'know' 'when' 'graduation' 'is'
  VP -> 'find' 'graduation' 'ceremony' 'date'
  VP -> 'work' 'as' 'a' 'student'
  VP -> 'get' 'help' 'with' 'internship'
  VP -> 'find' 'a' 'job' 'as' 'a' 'student'
  VP -> 'get' 'career' 'support'
  VP -> 'apply' 'for' 'work' 'study' 'program'
  VP -> 'request' 'services' 'as' 'an' 'alumni'
  VP -> 'can' 'alumni' 'access' 'student' 'services'
  VP -> 'can' 'former' 'students' 'get' 'services'
  VP -> 'what' 'services' 'are' 'available' 'for' 'alumni'
  VP -> 'find' 'housing' 'information'
  VP -> 'get' 'housing' 'information'
  VP -> 'access' 'housing' 'information'
  VP -> 'apply' 'for' 'housing'
  VP -> 'apply' 'for' 'accommodation'
  VP -> 'request' 'student' 'housing'
  VP -> 'find' 'student' 'accommodation'
  VP -> 'get' 'student' 'accommodation'
  VP -> 'know' 'where' 'to' 'stay' 'on' 'campus'
  VP -> 'find' 'the' 'library'
  VP -> 'borrow' 'books' 'from' 'the' 'library'
  VP -> 'visit' 'the' 'health' 'center'
  VP -> 'access' 'health' 'services'
  VP -> 'use' 'campus' 'health' 'center'
  VP -> 'visit' 'campus' 'clinic'
  VP -> 'go' 'to' 'the' 'health' 'center'
  VP -> 'get' 'medical' 'help' 'on' 'campus'
  VP -> 'see' 'a' 'doctor' 'on' 'campus'
  VP -> 'is' 'there' 'a' 'lost' 'and' 'found' 'on' 'campus'
  VP -> 'report' 'a' 'lost' 'item'
  VP -> 'find' 'a' 'lost' 'item'
  VP -> 'recover' 'my' 'lost' 'property'
  VP -> 'go' 'to' 'lost' 'and' 'found'
""")



category_map = {
    'apply for graduation': 'Graduation & Certification',
    'get a certificate': 'Graduation & Certification',
    'get my degree certificate': 'Graduation & Certification',
    'replace a lost student id': 'Lost & Found / ID',
    'collect my student id': 'Lost & Found / ID',
    'book an appointment with the dean': 'Appointments',
    'book an appointment with the advisor': 'Appointments',
    'join a student club': 'Clubs & Activities',
    'join a student organisation': 'Clubs & Activities',
    'join a club': 'Clubs & Activities',
    'join a organization': 'Clubs & Activities',
    'apply to be a student representative': 'Clubs & Activities',
    'speak to someone about academic counseling': 'Academic Support',
    'access my transcript': 'Transcripts & Results',
    'verify my academic records': 'Transcripts & Results',
    'get a letter of recommendation': 'Transcripts & Results',
    'get a student status letter': 'Letters & Certificates',
    'get a certificate of enrollment': 'Letters & Certificates',
    'register for my courses': 'Registration & Exams',
    'drop a course': 'Registration & Exams',
    'add a course': 'Registration & Exams',
    'miss an exam': 'Registration & Exams',
    'access past exam papers': 'Library & Academic Resources',
    'access academic resources': 'Library & Academic Resources',
    'get a fee structure': 'Finance & Fees',
    'pay my tuition fees': 'Finance & Fees',
    'pay in installments': 'Finance & Fees',
    'apply for a scholarship': 'Finance & Fees',
    'apply for a bursary': 'Finance & Fees',
    'get financial aid': 'Finance & Fees',
    'request a refund': 'Finance & Fees',
    'apply for a student loan': 'Finance & Fees',
    'get an invoice': 'Finance & Fees',
    'check if payment was received': 'Finance & Fees',
    'face penalties for late fee payment': 'Finance & Fees',
    'apply for academic leave': 'Registration & Exams',
    'defer my admission': 'Admissions',
    'apply to this university': 'Admissions',
    'submit my application documents': 'Admissions',
    'track my admission status': 'Admissions',
    'fix a mistake on my application': 'Admissions',
    'receive my admission letter': 'Admissions',
    'check graduation requirements': 'Graduation & Certification',
    'check when graduation ceremony takes place': 'Graduation & Certification',
    'work as a student': 'Career & Opportunities',
    'request services as an alumni': 'Administration',
    'find housing information': 'Housing & Campus Life',
    'find the library': 'Library & Academic Resources',
    'borrow books from the library': 'Library & Academic Resources',
    'visit the health center': 'Health Services',
    'access health services': 'Health Services',
    'where can i pay tuition fees':'Finance & Fees',
    'how do i pay my fees': 'Finance & Fees',
    'get a fee invoice': 'Finance & Fees',
    'how do i get my invoice': 'Finance & Fees',
    'can i pay fees in installments': 'Finance & Fees',
    'what financial aid options are available': 'Finance & Fees',
    "i paid but my payment isn't showing": 'Finance & Fees',
    'are there penalties for late fee payment': 'Finance & Fees',
    'can i get a refund if i withdraw': 'Finance & Fees',
    'can alumni access student services': 'Administration',
    'can former students get services': 'Administration',
    'what services are available for alumni': 'Administration',
    'get help with internship': 'Career & Opportunities',
    'find a job as a student': 'Career & Opportunities',
    'get career support': 'Career & Opportunities',
    'apply for work study program': 'Career & Opportunities',
    'get past papers': 'Library & Academic Resources',
    'get lecture notes': 'Library & Academic Resources',
    'download study materials': 'Library & Academic Resources',
    'where can i access academic resources': 'Library & Academic Resources',
    'confirm my academic records': 'Letters & Certificates',
    'check my academic records': 'Letters & Certificates',
    'are my records accurate': 'Letters & Certificates',
    'how can i verify my records': 'Letters & Certificates',
    'what if my name is misspelled on my certificate': 'Graduation & Certification',
    'correct my certificate': 'Graduation & Certification',
    'change my name on certificate': 'Graduation & Certification',
    'fix a mistake on my certificate': 'Graduation & Certification',
    'send transcripts to employers': 'Transcripts & Results',
    'send transcripts to other schools': 'Transcripts & Results',
    'are transcripts sent to institutions': 'Transcripts & Results',
    'know graduation requirements': 'Graduation & Certification',
    'view graduation requirements': 'Graduation & Certification',
    'see graduation requirements': 'Graduation & Certification',
    'what are the graduation requirements': 'Graduation & Certification',
    'when is the graduation ceremony': 'Graduation & Certification',
    'when will graduation happen': 'Graduation & Certification',
    'know when graduation is': 'Graduation & Certification',
    'find graduation ceremony date': 'Graduation & Certification',
    'collect my degree certificate': 'Graduation & Certification',
    'receive my degree certificate': 'Graduation & Certification',
    'claim my certificate': 'Graduation & Certification',
    'how do i get my certificate': 'Graduation & Certification',
    'request a letter of recommendation': 'Letters & Certificates',
    'apply for a letter of recommendation': 'Letters & Certificates',
    'ask for a letter of recommendation': 'Letters & Certificates',
    'get a letter confirming enrollment': 'Letters & Certificates',
    'request a student status letter': 'Letters & Certificates',
    'request a certificate of enrollment': 'Letters & Certificates',
    'apply for a student status letter': 'Letters & Certificates',
    'run for student representative': 'Clubs & Activities',
    'nominate myself as student representative': 'Clubs & Activities',
    'get elected as student leader': 'Clubs & Activities',
    'join student government': 'Clubs & Activities',
    'use campus health center': 'Health Services',
    'visit campus clinic': 'Health Services',
    'go to the health center': 'Health Services',
    'get medical help on campus': 'Health Services',
    'see a doctor on campus': 'Health Services',
    'is there a lost and found on campus': 'Lost & Found / ID',
    'report a lost item': 'Lost & Found / ID',
    'find a lost item': 'Lost & Found / ID',
    'recover my lost property': 'Lost & Found / ID',
    'go to lost and found': 'Lost & Found / ID',
    'book an appointment with the dean': 'Appointments',
    'book an appointment with the advisor': 'Appointments',
    'schedule a meeting with the dean': 'Appointments',
    'schedule a meeting with the advisor': 'Appointments',
    'arrange a meeting with the dean': 'Appointments',
    'arrange a meeting with the advisor': 'Appointments',
    'set up a meeting with the dean': 'Appointments',
    'set up a meeting with the advisor': 'Appointments',
    'speak to someone about academic counseling': 'Academic Support',
    'talk to someone about academic counseling': 'Academic Support',
    'get academic counseling': 'Academic Support',
    'meet an academic advisor': 'Academic Support',
    'meet with an academic advisor': 'Academic Support',
    'see an academic counselor': 'Academic Support',
    'get help with academic issues': 'Academic Support',
    'join a student club': 'Clubs & Activities',
    'join a student organization': 'Clubs & Activities',
    'join a campus club': 'Clubs & Activities',
    'join a campus organization': 'Clubs & Activities',
    'sign up for a student club': 'Clubs & Activities',
    'sign up for a student organization': 'Clubs & Activities',
    'participate in student clubs': 'Clubs & Activities',
    'participate in campus activities': 'Clubs & Activities',
    'become a member of a student club': 'Clubs & Activities',
    'become a member of a student organization': 'Clubs & Activities',
    'find housing information': 'Housing & Campus Life',
    'get housing information': 'Housing & Campus Life',
    'access housing information': 'Housing & Campus Life',
    'apply for housing': 'Housing & Campus Life',
    'apply for accommodation': 'Housing & Campus Life',
    'request student housing': 'Housing & Campus Life',
    'find student accommodation': 'Housing & Campus Life',
    'get student accommodation': 'Housing & Campus Life',
    'know where to stay on campus': 'Housing & Campus Life',
    'check my final results': 'Transcripts & Results',
    'see my final results': 'Transcripts & Results',
    'view my final results': 'Transcripts & Results',
    'get my final results': 'Transcripts & Results',
    'access my final results': 'Transcripts & Results',
    'find my final results': 'Transcripts & Results',
    'when will my final results be released': 'Transcripts & Results',
    'know when final results are released': 'Transcripts & Results'
    
}
    



response_map = {
    'Transcripts & Results': "Final results are released through the academic portal at the end of each semester.",
    'Graduation & Certification': "Graduation details, including requirements and certificate collection, are available on the university portal or through the registrar's office. For name corrections, contact the registrar immediately with official documentation.",
    'Lost & Found / ID': "Lost and found services are located at the campus security office. Please report or check for lost items there.",
    'Appointments': "To book an appointment with the dean or advisor, please visit the department office or use the university’s online appointment scheduling portal.",
    'Clubs & Activities': "You can join clubs and student organizations during orientation week or by visiting the student affairs office. Look out for club registration drives and events on campus.",
    'Academic Support': "You can speak to an academic advisor or counselor at your department office or through the student support center for guidance on academic matters.",
    'Letters & Certificates': "To get a recommendation letter or enrollment certificate, contact the registrar’s office or apply via the academic portal.",
    'Registration & Exams': "You can register or make changes to your courses during the registration window on the portal.",
    'Library & Academic Resources': "You can access past papers, notes, and other academic materials through the university library portal or academic department websites.",
    'Finance & Fees': "Tuition fees can be paid through the finance portal or at the university cashier's office. Payment plans, invoices, and refunds are available. Contact the finance office for help with payment issues or financial aid.",
    'Admissions': "For admissions, track your application or submit documents online. You can also defer if needed.",
    'Career & Opportunities': "The university provides internship and career services through the career office. You can also apply for work-study opportunities during registration.",
    'Administration': "Alumni can request limited services such as transcripts and verification letters through the alumni office or registrar.",
    'Housing & Campus Life': "Housing and accommodation information is available at the student services or housing office. You can also apply through the university portal.",
    'Health Services': "The campus health center is available to all students for medical services. You can visit the clinic during working hours or schedule an appointment."
}
# Initialize the parser
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
