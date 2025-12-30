#!/usr/bin/env python3
"""
Test script for multi-variant detection functionality.
Tests the parsing, scoring, and title-based selection logic.
"""

import re
import sys

# ========== HELPER FUNCTIONS (copied from renommeur.py) ==========

def simplify_institution_name(name):
    """Supprime articles/formes juridiques en tête et fin pour garder un nom court."""
    if not name:
        return name
    cleaned = name.strip()
    # Supprimer articles au début
    cleaned = re.sub(r"^(la|le|les|l'|the)\s+", "", cleaned, flags=re.IGNORECASE)
    # Supprimer formes juridiques (avec ou sans points): S.A., S.A.S., SA, SAS, SARL, etc.
    cleaned = re.sub(
        r"\s+(s\.?a\.?(?:s\.?)?|sarl|scs|snc|sca|gmbh|inc\.?|ltd\.?|plc|llc|corp\.?|company|limited|anonyme|soci[eé]t[eé])\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or name.strip()

def title_from_first_page(first_page_text):
    """Heuristique : premier intitulé plausible sur la 1ère page."""
    if not first_page_text:
        return None
    for line in first_page_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if len(candidate) < 6 or len(candidate) > 80:
            continue
        if sum(ch.isalpha() for ch in candidate) < max(5, len(candidate) // 3):
            continue
        if re.match(r"^(page|document|annexe|table(au)?|index|sommaire)", candidate, re.IGNORECASE):
            continue
        return candidate
    return None

def best_match_with_title(variants, title_text):
    """Sélectionne la variante la plus proche du titre du document."""
    if not title_text:
        return variants[0] if variants else "inconnu"
    
    # Normaliser le titre
    title_norm = title_text.lower().strip()
    
    # Scorer chaque variante
    best_variant = variants[0]
    best_score = 0
    
    for variant in variants:
        if variant.lower() == "inconnu":
            continue
        
        variant_norm = variant.lower().strip()
        
        # Score basé sur la similarité avec le titre
        # Plus il y a de mots en commun, meilleur le score
        title_words = set(title_norm.split())
        variant_words = set(variant_norm.split())
        common_words = len(title_words & variant_words)
        
        # Bonus si c'est exactement le titre
        if variant_norm == title_norm:
            score = 1000
        # Bonus si c'est un sous-ensemble du titre
        elif variant_norm in title_norm:
            score = 500 + common_words * 50
        # Sinon, compter les mots en commun
        else:
            score = common_words * 50
        
        if score > best_score:
            best_score = score
            best_variant = variant
    
    return best_variant

def parse_analysis(text, first_page_text=None):
    """Parse la réponse Ollama avec 3 variantes."""
    if not text:
        return None, None, None, False
    
    inst_variants = []
    obj_variants = []
    date = "inconnu"
    
    # Parser strictement chaque ligne
    for line in text.splitlines():
        line_lower = line.lower()
        
        # Institution
        if line_lower.startswith("institution variante"):
            value = line.split(":", 1)[1].strip() if ":" in line else ""
            value = re.sub(r'\s*[\(\[].*$', '', value).strip()
            if value:
                inst_variants.append(value)
        
        # Objet
        elif line_lower.startswith("objet variante"):
            value = line.split(":", 1)[1].strip() if ":" in line else ""
            value = re.sub(r'\s*[\(\[].*$', '', value).strip()
            if value:
                obj_variants.append(value)
        
        # Date
        elif line_lower.startswith("date:"):
            value = line.split(":", 1)[1].strip()
            value = re.sub(r'\s*[\(\[].*$', '', value).strip()
            if re.match(r"^\d{4}-\d{2}$", value):
                date = value
            elif value.lower() != "inconnu":
                match = re.search(r"(\d{4})-(\d{2})", value)
                if match:
                    date = f"{match.group(1)}-{match.group(2)}"
    
    # Assurer au moins 3 variantes (remplir avec "inconnu")
    while len(inst_variants) < 3:
        inst_variants.append("inconnu")
    while len(obj_variants) < 3:
        obj_variants.append("inconnu")
    
    # Sélectionner la meilleure variante basée sur le titre
    title = title_from_first_page(first_page_text)
    inst = best_match_with_title(inst_variants[:3], title)
    obj = best_match_with_title(obj_variants[:3], title)
    
    # Simplifier institution
    inst = simplify_institution_name(inst)
    
    # Validation
    if date == "inconnu" or not re.match(r"^\d{4}-\d{2}$", date):
        return inst, obj, date, False
    
    # Max 1 champ inconnu
    unknown_count = sum(1 for v in [inst, obj] if v == "inconnu")
    certitude = unknown_count <= 1
    return inst, obj, date, certitude

# ========== TEST CASES ==========

def test_parse_analysis():
    """Test parsing of multi-variant Ollama output."""
    print("=" * 60)
    print("TEST 1: Parse Analysis with 3 Institution variants")
    print("=" * 60)
    
    ollama_output = """Institution Variante 1: Banque De France
Institution Variante 2: Banque Nationale
Institution Variante 3: inconnu
Objet Variante 1: Releve De Compte
Objet Variante 2: Releve Bancaire
Objet Variante 3: Document Financier
Date: 2024-01"""
    
    first_page = """Relevé de compte
Janvier 2024
Banque De France"""
    
    inst, obj, date, certitude = parse_analysis(ollama_output, first_page)
    print(f"Institution: {inst}")
    print(f"Objet: {obj}")
    print(f"Date: {date}")
    print(f"Certitude: {certitude}")
    print(f"✓ PASS" if inst == "Banque De France" and obj == "Releve De Compte" else "✗ FAIL")
    print()

def test_title_extraction():
    """Test title extraction heuristics."""
    print("=" * 60)
    print("TEST 2: Title Extraction from First Page")
    print("=" * 60)
    
    # Valid title
    first_page1 = """Relevé de compte
Janvier 2024
Banque De France

Solde: 1000 EUR"""
    
    title1 = title_from_first_page(first_page1)
    print(f"Test 2a - Valid title: {title1}")
    print(f"✓ PASS" if title1 == "Relevé de compte" else "✗ FAIL")
    
    # Ignore headers
    first_page2 = """Page 1
Table des matières
Facture du mois de janvier
"""
    
    title2 = title_from_first_page(first_page2)
    print(f"Test 2b - Skip headers: {title2}")
    print(f"✓ PASS" if title2 == "Facture du mois de janvier" else "✗ FAIL")
    
    # Too short
    first_page3 = """No
Valid content"""
    
    title3 = title_from_first_page(first_page3)
    print(f"Test 2c - Too short: {title3}")
    print(f"✓ PASS" if title3 == "Valid content" else "✗ FAIL")
    print()

def test_best_match():
    """Test variant selection based on title."""
    print("=" * 60)
    print("TEST 3: Best Match Selection with Title")
    print("=" * 60)
    
    # Test 3a: Exact match
    variants = ["Banque De France", "Société Générale", "inconnu"]
    title = "Banque De France"
    result = best_match_with_title(variants, title)
    print(f"Test 3a - Exact match: {result}")
    print(f"✓ PASS" if result == "Banque De France" else "✗ FAIL")
    
    # Test 3b: Substring match
    variants = ["Relevé De Compte", "Relevé Bancaire", "Document Financier"]
    title = "Relevé de compte bancaire"
    result = best_match_with_title(variants, title)
    print(f"Test 3b - Substring match: {result}")
    print(f"✓ PASS" if result == "Relevé De Compte" else "✗ FAIL")
    
    # Test 3c: Word overlap
    variants = ["Contrat De Travail", "Fiche De Paie", "inconnu"]
    title = "Contrat emploi 2024"
    result = best_match_with_title(variants, title)
    print(f"Test 3c - Word overlap: {result}")
    print(f"✓ PASS" if result == "Contrat De Travail" else "✗ FAIL")
    
    # Test 3d: All unknown
    variants = ["inconnu", "inconnu", "inconnu"]
    title = "Facture"
    result = best_match_with_title(variants, title)
    print(f"Test 3d - All unknown: {result}")
    print(f"✓ PASS" if result == "inconnu" else "✗ FAIL")
    print()

def test_simplify_institution():
    """Test institution name simplification."""
    print("=" * 60)
    print("TEST 4: Institution Name Simplification")
    print("=" * 60)
    
    test_cases = [
        ("La Banque Nationale S.A.", "Banque Nationale"),
        ("Le Crédit Mutuel S.A.S.", "Crédit Mutuel"),
        ("Orange S.A.", "Orange"),
        ("La Société Générale", "Société Générale"),
        ("inconnu", "inconnu"),
    ]
    
    for input_name, expected in test_cases:
        result = simplify_institution_name(input_name)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_name}' → '{result}' (expected: '{expected}')")
    print()

def test_missing_variants():
    """Test parsing when fewer than 3 variants returned."""
    print("=" * 60)
    print("TEST 5: Handle Missing Variants (Padding)")
    print("=" * 60)
    
    # Only 1 variant provided by Ollama
    ollama_output = """Institution Variante 1: Banque De France
Objet Variante 1: Releve De Compte
Date: 2024-01"""
    
    inst, obj, date, certitude = parse_analysis(ollama_output, None)
    print(f"Institution: {inst} (should have 1 valid + 2 padded)")
    print(f"Objet: {obj} (should have 1 valid + 2 padded)")
    print(f"Date: {date}")
    print(f"Certitude: {certitude} (should be True)")
    print(f"✓ PASS" if inst == "Banque De France" and certitude else "✗ FAIL")
    print()

def test_invalid_date():
    """Test parsing with invalid date."""
    print("=" * 60)
    print("TEST 6: Invalid Date Handling")
    print("=" * 60)
    
    ollama_output = """Institution Variante 1: Banque De France
Institution Variante 2: inconnu
Institution Variante 3: inconnu
Objet Variante 1: Releve
Objet Variante 2: inconnu
Objet Variante 3: inconnu
Date: janvier 2024"""
    
    inst, obj, date, certitude = parse_analysis(ollama_output, None)
    print(f"Institution: {inst}")
    print(f"Objet: {obj}")
    print(f"Date: {date}")
    print(f"Certitude: {certitude} (should be False - invalid date)")
    print(f"✓ PASS" if not certitude else "✗ FAIL")
    print()

# ========== MAIN ==========

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-VARIANT DETECTION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_parse_analysis()
        test_title_extraction()
        test_best_match()
        test_simplify_institution()
        test_missing_variants()
        test_invalid_date()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
