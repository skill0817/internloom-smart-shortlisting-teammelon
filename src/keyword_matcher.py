"""
keyword_matcher.py
==================
Handles keyword-based skill extraction and matching between a Job Description
and a Candidate's Resume.
Key Features:
1. Canonical Skill Dictionary: Maps skill synonyms/aliases (e.g. 'JS' -> 'JavaScript').
2. Robust Boundary Matching: Accurately matches skills without partial-word false positives.
3. Required vs. Preferred Weighting: Allows required skills to carry more weight than nice-to-haves.
4. Clean, beginner-friendly functions with full explanations.
"""
import re
from typing import Dict, List, Set, Tuple, Any
# ==============================================================================
# CONFIGURATION & WEIGHTS
# Adjust these constants to change how the keyword score is calculated.
# ==============================================================================
REQUIRED_SKILLS_WEIGHT = 0.70   # 70% of keyword score comes from required skills
PREFERRED_SKILLS_WEIGHT = 0.30  # 30% of keyword score comes from preferred skills
# ==============================================================================
# CANONICAL SKILL VOCABULARY
# Format:
# "Canonical Skill Name": ["alias1", "alias2", ...]
# Teammates can easily add new skills and aliases here at any time.
# ==============================================================================
DEFAULT_SKILL_VOCABULARY: Dict[str, List[str]] = {
    # Programming Languages
    "Python": ["python", "py"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", "c sharp"],
    "Go": ["golang", "go"],
    "Ruby": ["ruby"],
    "PHP": ["php"],
    "Swift": ["swift"],
    "Kotlin": ["kotlin"],
    "Rust": ["rust"],
    "SQL": ["sql"],
    # Frontend Frameworks & Libraries
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angular.js", "angularjs"],    "Spring Boot": ["spring boot", "springboot", "spring"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Tailwind CSS": ["tailwind", "tailwindcss", "tailwind css"],
    "Bootstrap": ["bootstrap"],
    "Next.js": ["next.js", "nextjs", "next"],
    # Backend Frameworks
    "Node.js": ["node.js", "nodejs", "node"],
    "Express": ["express", "express.js", "expressjs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast-api"],
    "Spring Boot": ["spring boot", "springboot", "spring"],
    "ASP.NET": ["asp.net", "aspnet", ".net"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "SQLite": ["sqlite"],
    "Oracle": ["oracle db", "oracle database", "oracle"],
    # Cloud & DevOps
    "Docker": ["docker", "containerization", "containers"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Git": ["git", "github", "gitlab", "version control"],
    "CI/CD": ["ci/cd", "ci-cd", "cicd", "continuous integration"],
    "Linux": ["linux", "unix", "bash"],
    # AI / Data Science
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "dl"],
    "NLP": ["nlp", "natural language processing"],
    "TensorFlow": ["tensorflow", "tf"],
    "PyTorch": ["pytorch", "torch"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-Learn": ["scikit-learn", "sklearn"],
    # Architecture & Concepts
    "REST API": ["rest api", "rest apis", "restful api", "restful apis", "rest"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice architecture"],
    "Agile": ["agile", "scrum", "kanban"]
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _build_skill_regex(alias: str) -> re.Pattern:
    """
    Creates a regex pattern that ensures we match the alias as a distinct word
    or term, avoiding false positives (e.g. 'c' matching inside 'cat', or 'js' in 'json').
    
    We use negative lookbehind (?<![a-zA-Z0-9]) and negative lookahead (?![a-zA-Z0-9])
    to safely handle special characters like '+', '#', and '.'.
    """
    escaped_alias = re.escape(alias)
    pattern = rf"(?<![a-zA-Z0-9]){escaped_alias}(?![a-zA-Z0-9])"
    return re.compile(pattern, re.IGNORECASE)
def find_skills_in_text(
    text: str,
    skill_vocab: Dict[str, List[str]] = None
) -> Set[str]:
    """
    Scans a given text (job description or resume) and returns the set of
    CANONICAL skill names found.
    
    Example:
        If text has 'NodeJS' and 'JS', this returns {'Node.js', 'JavaScript'}.
    """
    if not text or not text.strip():
        return set()


    if skill_vocab is None:
        skill_vocab = DEFAULT_SKILL_VOCABULARY
    found_skills: Set[str] = set()
    for canonical_name, aliases in skill_vocab.items():
        # Check all possible names/aliases for this skill
        all_variants = [canonical_name] + aliases
        for variant in all_variants:
            pattern = _build_skill_regex(variant)
            if pattern.search(text):
                found_skills.add(canonical_name)
                break  # Found this canonical skill, no need to check other aliases
    return found_skills
def extract_skills_by_section(
    job_description_text: str,
    skill_vocab: Dict[str, List[str]] = None
) -> Tuple[Set[str], Set[str]]:
    """
    Splits the Job Description into 'Required' and 'Preferred' sections
    based on common section headers.
    
    Returns:
        (required_skills_set, preferred_skills_set)
        
    If no clear sections exist, all detected skills are considered 'Required'
    to ensure fairness and prevent unpenalized missing skills.
    """
    if not job_description_text or not job_description_text.strip():
        return set(), set()
    if skill_vocab is None:
        skill_vocab = DEFAULT_SKILL_VOCABULARY
    # Normalize line breaks
    text = job_description_text.replace("\r\n", "\n")

    # Define regex markers for required vs. preferred sections
    required_keywords = r"(?:required|requirements|must have|qualifications|minimum qualifications|what you need)"
    preferred_keywords = r"(?:preferred|nice to have|good to have|bonus|desired|plus|optional)"
    # Split text into lines to identify sections
    lines = text.split("\n")
    required_lines: List[str] = []
    preferred_lines: List[str] = []
    current_section = "other"  # Can be 'required', 'preferred', or 'other'
    for line in lines:
        stripped = line.strip().lower()
        # Check if line looks like a header
        if re.search(rf"\b{preferred_keywords}\b", stripped):
            current_section = "preferred"
            preferred_lines.append(line)
        elif re.search(rf"\b{required_keywords}\b", stripped):
            current_section = "required"
            required_lines.append(line)
        else:
            if current_section == "required":
                required_lines.append(line)
            elif current_section == "preferred":
                preferred_lines.append(line)
    required_text = "\n".join(required_lines)
    preferred_text = "\n".join(preferred_lines)
    required_skills = find_skills_in_text(required_text, skill_vocab)
    preferred_skills = find_skills_in_text(preferred_text, skill_vocab)

    # Preferred skills should not overlap with required skills
    preferred_skills = preferred_skills - required_skills
    # If section parsing did not detect any skills (e.g. no explicit headers used),
    # fallback to scanning the entire text and treating all found skills as required.
    if not required_skills and not preferred_skills:
        all_skills = find_skills_in_text(job_description_text, skill_vocab)
        return all_skills, set()
    return required_skills, preferred_skills

# ==============================================================================
# MAIN PUBLIC MATCHING FUNCTION
# ==============================================================================
def match_keywords(
    job_description_text: str,
    resume_text: str,
    skill_vocab: Dict[str, List[str]] = None,
    required_weight: float = REQUIRED_SKILLS_WEIGHT,
    preferred_weight: float = PREFERRED_SKILLS_WEIGHT
) -> Dict[str, Any]:
    """
    Compares a candidate's resume against a job description using keyword matching.
    
    Parameters:
        job_description_text (str): Full text of the job description.
        resume_text (str): Full text extracted from candidate's resume.
        skill_vocab (dict, optional): Custom skill vocabulary if desired.
        required_weight (float): Importance weight of required skills (default 0.70).
        preferred_weight (float): Importance weight of preferred skills (default 0.30).
        
    Returns:
        dict:
            {
                "matched_skills": sorted list of all matched skills,
                "missing_skills": sorted list of all missing skills,
                "keyword_score": float score from 0.0 to 100.0,
                "matched_required_skills": list of matched required skills,
                "missing_required_skills": list of missing required skills,
                "matched_preferred_skills": list of matched preferred skills,
                "missing_preferred_skills": list of missing preferred skills
            }
    """
    # Defensive checks for empty or non-string inputs
    if not isinstance(job_description_text, str) or not job_description_text.strip():
        return {
            "matched_skills": [],
            "missing_skills": [],
            "keyword_score": 0.0,
            "matched_required_skills": [],
            "missing_required_skills": [],
            "matched_preferred_skills": [],
            "missing_preferred_skills": []
        }

    if not isinstance(resume_text, str) or not resume_text.strip():
        # Job description has skills, but candidate resume is empty
        required_skills, preferred_skills = extract_skills_by_section(
            job_description_text, skill_vocab
        )
        all_missing = sorted(list(required_skills | preferred_skills))
        return {
            "matched_skills": [],
            "missing_skills": all_missing,
            "keyword_score": 0.0,
            "matched_required_skills": [],
            "missing_required_skills": sorted(list(required_skills)),
            "matched_preferred_skills": [],
            "missing_preferred_skills": sorted(list(preferred_skills))
        }
    # Step 1: Extract required and preferred skills from the Job Description
    required_jd_skills, preferred_jd_skills = extract_skills_by_section(
        job_description_text, skill_vocab
    )
    # Step 2: Extract all skills present in the candidate's resume
    resume_skills = find_skills_in_text(resume_text, skill_vocab)
    # Step 3: Compute intersections (matched) and differences (missing)
    matched_required = required_jd_skills.intersection(resume_skills)
    missing_required = required_jd_skills - resume_skills
    matched_preferred = preferred_jd_skills.intersection(resume_skills)
    missing_preferred = preferred_jd_skills - resume_skills


    # Step 4: Calculate weighted keyword score
    # Score for required skills (0.0 to 1.0)
    if len(required_jd_skills) > 0:
        req_ratio = len(matched_required) / len(required_jd_skills)
    else:
        req_ratio = 1.0  # If no required skills were specified, don't penalize
    # Score for preferred skills (0.0 to 1.0)
    if len(preferred_jd_skills) > 0:
        pref_ratio = len(matched_preferred) / len(preferred_jd_skills)
    else:
        pref_ratio = 1.0  # If no preferred skills were specified, don't penalize
    # Combine using weights
    if len(required_jd_skills) > 0 and len(preferred_jd_skills) > 0:
        # Both required and preferred exist: use full weighted combination
        final_ratio = (req_ratio * required_weight) + (pref_ratio * preferred_weight)
    elif len(required_jd_skills) > 0:
        # Only required skills exist: score is 100% based on required skills
        final_ratio = req_ratio
    elif len(preferred_jd_skills) > 0:
        # Only preferred skills exist: score is 100% based on preferred skills
        final_ratio = pref_ratio
    else:
        # Neither required nor preferred skills detected in the JD
        final_ratio = 0.0
    # Scale to 0 - 100
    keyword_score = round(final_ratio * 100.0, 1)
    all_matched = sorted(list(matched_required | matched_preferred))
    all_missing = sorted(list(missing_required | missing_preferred))

    return {
        "matched_skills": all_matched,
        "missing_skills": all_missing,
        "keyword_score": keyword_score,
        "matched_required_skills": sorted(list(matched_required)),
        "missing_required_skills": sorted(list(missing_required)),
        "matched_preferred_skills": sorted(list(matched_preferred)),
        "missing_preferred_skills": sorted(list(missing_preferred))
    }