"""
Utilities for detecting and anonymizing personally identifiable information (PII).
"""

import re
from collections import defaultdict


def detect_pii(text):
    """
    Detects potential PII (Personally Identifiable Information) in text.
    
    Args:
        text (str): The text to check for PII
        
    Returns:
        dict: Dictionary with PII type as keys and a list of tuples 
              (start, end, value) for each match
    """
    patterns = {
        # Improved email pattern with more TLDs and special characters
        "email": r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b',
        
        # Enhanced phone pattern to catch more international formats
        "phone": r'\b(?:(?:\+|00)\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?)?(?:\d{3,4}[ -]?)(?:\d{3,4})\b',
        
        # PESEL with optional validation context (11 digits)
        "pesel": r'\b(?:PESEL:?\s*)?\d{11}\b',
        
        # Improved address pattern for various formats
        "address": r'\b(?:ul\.|ulica|al\.|aleja|os\.|osiedle|pl\.|plac)\s+[^,\n]{2,40}(?:[ ,]+(?:\d+[a-zA-Z]?(?:\/\d+[a-zA-Z]?)?)?)\b',
        
        # Enhanced NIP pattern including optional 'NIP' prefix
        "nip": r'\b(?:NIP:?\s*)?\d{3}[-]?\d{3}[-]?\d{2}[-]?\d{2}\b',
        
        # REGON pattern including optional 'REGON' prefix (9 or 14 digits)
        "regon": r'\b(?:REGON:?\s*)?(?:\d{9}|\d{14})\b',
        
        # Improved name pattern supporting more naming conventions and accented characters
        "name_surname": r'\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:[ -][A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)+\b',
        
        # National ID (Dowód osobisty)
        "id_card": r'\b[A-Z]{3}\s?\d{6}\b',
        
        # Passport number
        "passport": r'\b[A-Z]{2}\s?\d{7}\b',
        
        # Credit card pattern
        "credit_card": r'\b(?:\d{4}[ -]?){3}\d{4}\b',
        
        # Bank account number (Polish format)
        "bank_account": r'\b\d{2}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b'
    }
    
    findings = {}
    for pii_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            if pii_type not in findings:
                findings[pii_type] = []
            findings[pii_type].append((match.start(), match.end(), match.group()))
    
    return findings


def anonymize_text(text, pii_findings=None, consistent=True):
    """
    Anonymizes PII in text by replacing with placeholders.
    
    Args:
        text (str): The text to anonymize
        pii_findings (dict, optional): Dictionary of PII findings from detect_pii
            If None, detect_pii will be called
        consistent (bool): Whether to replace same values with consistent identifiers
    
    Returns:
        str: Anonymized text
    """
    if pii_findings is None:
        pii_findings = detect_pii(text)
    
    # Sort findings from end to start to avoid changing indexes when replacing
    replacements = []
    
    # For consistent anonymization, keep track of values already seen
    value_mappings = defaultdict(dict)
    counter = defaultdict(int)
    
    for pii_type, findings in pii_findings.items():
        for start, end, value in findings:
            if consistent:
                # If we've seen this value before, use the same replacement
                if value not in value_mappings[pii_type]:
                    counter[pii_type] += 1
                    value_mappings[pii_type][value] = counter[pii_type]
                
                identifier = value_mappings[pii_type][value]
                replacement = f"[{pii_type.upper()}-{identifier}]"
            else:
                replacement = f"[{pii_type.upper()}]"
                
            replacements.append((start, end, value, replacement))
    
    # Sort replacements from end to start to preserve text indexes
    replacements.sort(reverse=True, key=lambda x: x[0])
    
    # Replace PII with placeholders
    anonymized_text = text
    for start, end, value, replacement in replacements:
        anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
    
    return anonymized_text


def batch_anonymize_text(texts, consistent_across_texts=True):
    """
    Anonymizes PII across multiple texts with consistent replacements.
    
    Args:
        texts (list): List of text strings to anonymize
        consistent_across_texts (bool): Whether to use consistent replacements across texts
    
    Returns:
        list: List of anonymized texts
    """
    if not consistent_across_texts:
        return [anonymize_text(text) for text in texts]
    
    # For consistent anonymization across texts, first collect all PII
    global_value_mappings = defaultdict(dict)
    global_counter = defaultdict(int)
    all_findings = []
    
    # First pass: detect PII in all texts
    for text in texts:
        findings = detect_pii(text)
        all_findings.append(findings)
        
        # Build global mapping of values to identifiers
        for pii_type, matches in findings.items():
            for _, _, value in matches:
                if value not in global_value_mappings[pii_type]:
                    global_counter[pii_type] += 1
                    global_value_mappings[pii_type][value] = global_counter[pii_type]
    
    # Second pass: anonymize each text using the global mappings
    anonymized_texts = []
    for i, text in enumerate(texts):
        findings = all_findings[i]
        replacements = []
        
        for pii_type, matches in findings.items():
            for start, end, value in matches:
                identifier = global_value_mappings[pii_type][value]
                replacement = f"[{pii_type.upper()}-{identifier}]"
                replacements.append((start, end, value, replacement))
        
        # Sort replacements from end to start
        replacements.sort(reverse=True, key=lambda x: x[0])
        
        # Apply replacements
        anonymized = text
        for start, end, _, replacement in replacements:
            anonymized = anonymized[:start] + replacement + anonymized[end:]
        
        anonymized_texts.append(anonymized)
    
    return anonymized_texts