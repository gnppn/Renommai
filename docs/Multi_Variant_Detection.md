# Multi-Variant Detection Implementation

## Overview

The Renommai document renaming system has been enhanced with **multi-variant detection**. Instead of Ollama providing a single Institution and Objet value, it now generates **3 candidates for each field** (Institution and Objet), allowing the system to intelligently select the best match based on the document's title.

## Motivation

- **Improved Accuracy**: Multiple candidates increase the chance that at least one is correct
- **Title-Based Selection**: The system can now compare variants against the document's detected title to pick the best match
- **Better Confidence**: If all 3 variants are "unknown" or irrelevant, the system flags the file as uncertain
- **Flexible Parsing**: Handles cases where Ollama returns fewer than 3 variants (fills with "unknown")

## Changes in Both Scripts

### 1. Updated PROMPT_TEMPLATE

**New Format**:
```
Institution Variante 1: <most probable>
Institution Variante 2: <less probable>
Institution Variante 3: <least probable>
Objet Variante 1: <most probable>
Objet Variante 2: <less probable>
Objet Variante 3: <least probable>
Date: <YYYY-MM>
```

**Example Output from Ollama**:
```
Institution Variante 1: Banque De France
Institution Variante 2: Banque Nationale
Institution Variante 3: inconnu
Objet Variante 1: Releve De Compte
Objet Variante 2: Releve Bancaire
Objet Variante 3: Document Financier
Date: 2024-01
```

### 2. New Helper Functions

#### `title_from_first_page(first_page_text)`
- Extracts the first plausible title from the document's first page
- Heuristics applied:
  - Candidate must be 6-80 characters long
  - Must contain sufficient alphabetic characters
  - Excludes generic headers (page, document, table, etc.)
- Returns: `None` if no title found, otherwise the title string

#### `best_match_with_title(variants, title_text)`
- Compares 3 variants against the document's title
- Scoring algorithm:
  1. **Exact match**: Score 1000
  2. **Substring match**: Score 500 + word_overlap × 50
  3. **Word overlap**: Score word_overlap × 50
- Returns: The variant with the highest score, or the first variant if no title available

#### `simplify_institution_name(name)`
- Removes articles (la, le, les, l') and legal forms (S.A., S.A.S., SARL, etc.)
- Applied to the selected Institution variant
- Example: "La Banque Nationale S.A.S." → "Banque Nationale"

### 3. Updated `parse_analysis()` Function

**Signature**:
```python
def parse_analysis(text, first_page_text=None):
    """Parse Ollama response with 3 variants."""
    # Returns: (institution, objet, date, certitude)
```

**Logic**:
1. Parse all 7 lines from Ollama output (3 Institution variants + 3 Objet variants + Date)
2. Extract the document's title from first page
3. Use `best_match_with_title()` to select the best Institution and Objet variants
4. Apply `simplify_institution_name()` to the selected institution
5. Validate date format (YYYY-MM)
6. Calculate `certitude` flag:
   - `True`: Date valid AND at most 1 field is "unknown"
   - `False`: Otherwise

**Return values**:
- `institution`: Selected Institution variant (or "unknown")
- `objet`: Selected Objet variant (or "unknown")
- `date`: Extracted date (or "unknown")
- `certitude`: Boolean flag indicating analysis confidence

### 4. Script-Specific Changes

#### `renommeur.py`
- **Line 450-452**: `generate_name()` uses Title Case formatting
- **Line 540-555**: `simplify_institution_name()` implementation
- **Line 556-572**: `title_from_first_page()` implementation
- **Line 573-610**: `best_match_with_title()` implementation
- **Line 615-665**: `parse_analysis()` with multi-variant handling

#### `renommeur_interactif.py`
- **Line 401-440**: Updated PROMPT_TEMPLATE with multi-variant format
- **Line 462-530**: New helper functions + updated `parse_analysis()`
- **Line 720-726**: Call to `parse_analysis()` with `first_page_text` parameter

## Backward Compatibility

- Both scripts remain fully functional with the updated Ollama prompt
- If Ollama returns fewer than 3 variants, the functions pad the list with "unknown"
- If `first_page_text` is not provided, `best_match_with_title()` defaults to Variant 1

## Testing Recommendations

1. **Test with diverse documents**:
   - Multiple institutions (bank, insurance, employer, etc.)
   - Various document types (invoices, statements, contracts)

2. **Verify title extraction**:
   - Ensure `title_from_first_page()` correctly identifies document titles
   - Check that generic headers are properly filtered

3. **Validate variant selection**:
   - Confirm that the system selects sensible variants when multiple are provided
   - Test edge cases (all "unknown", missing variants, etc.)

4. **Monitor prompt stability**:
   - Llama3 should consistently return 3 variants
   - Watch for formatting deviations that break the parser

## Performance Impact

- **Minimal CPU overhead**: Title extraction and scoring are lightweight
- **Context usage**: Prompt increased slightly (~100 tokens) but still within 4K window
- **Analysis time**: No noticeable difference (Ollama dominates timing)

## Future Enhancements

- Add confidence scores for each variant
- Implement fuzzy matching for title comparison
- Allow user-defined title patterns
- Add variant confidence visualization in interactive mode
