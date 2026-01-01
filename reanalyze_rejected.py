#!/usr/bin/env python3
"""Re-analyse automatiquement les fichiers présents dans les dossiers Echec_*.

But: non-interactive, permissive: force OCR lang fra+eng, accept low certitude.
"""
import os
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path.cwd()

def find_echec_dirs(root):
    return [p for p in root.glob('**/Echec_*') if p.is_dir()]

def main():
    from renommeur_simplifié import (
        create_searchable_pdf_from_original,
        analyze_vision,
        extract_dates,
        analyze_ollama,
        parse_analysis,
        generate_name,
        enrich_pdf_with_vision_text,
        load_config,
    )
    cfg = load_config()
    model = cfg.get('OLLAMA_MODEL')

    echec_dirs = find_echec_dirs(ROOT)
    if not echec_dirs:
        print('No Echec_* directories found.')
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []

    for d in echec_dirs:
        out_dir = d.parent / f"RERUN_{d.name}_{timestamp}"
        out_dir.mkdir(exist_ok=True)
        print(f'Processing directory: {d} -> {out_dir}')

        for f in d.iterdir():
            if not f.is_file():
                continue
            # prefer original extensions
            if f.suffix.lower() not in ['.pdf','.png','.jpg','.jpeg','.docx','.xlsx']:
                continue

            print('\n-- File:', f.name)
            try:
                text_primary, tmp_pdf = create_searchable_pdf_from_original(str(f))
            except Exception as e:
                print('  create_searchable_pdf_from_original failed:', e)
                text_primary, tmp_pdf = None, None

            # prepare image for vision
            vision_text = ''
            vision_struct = ''
            image_for_vision = None
            if f.suffix.lower() in ['.png','.jpg','.jpeg']:
                image_for_vision = str(f)
            elif tmp_pdf:
                # extract first page image using pdfplumber
                try:
                    import pdfplumber
                    with pdfplumber.open(tmp_pdf) as pdf:
                        if pdf.pages:
                            img = pdf.pages[0].to_image(resolution=300).original
                            if img:
                                t = out_dir / (f.stem + '_page1.png')
                                img.save(t)
                                image_for_vision = str(t)
                except Exception:
                    image_for_vision = None

            if image_for_vision:
                try:
                    av = analyze_vision(image_for_vision)
                    if isinstance(av, dict):
                        vision_text = av.get('text','')
                        vision_struct = av.get('structured','')
                except Exception as e:
                    print('  analyze_vision failed:', e)

            # Combine texts for date extraction
            combined = ''
            if vision_text:
                combined = vision_text + '\n' + (text_primary or '')
            else:
                combined = text_primary or ''

            dates = extract_dates(combined)
            print('  dates:', dates)

            # Run Ollama as fallback (non-interactive)
            analysis = None
            try:
                analysis = analyze_ollama(text_primary or combined, dates, model, vision_analysis=vision_text, pass_level='reanalyze', original_filename=f.name)
            except Exception as e:
                print('  analyze_ollama failed:', e)

            inst, obj, date, cert = parse_analysis(analysis or vision_struct, text_primary)

            # Permissive: accept even if cert False; if date invalid, keep 'inconnu'
            if not date or not date.startswith('20'):
                date = date if date and date.startswith('20') else 'inconnu'

            newname = generate_name(inst or 'inconnu', obj or 'inconnu', date, f.suffix)
            # Save outputs into out_dir
            try:
                target = out_dir / newname
                if tmp_pdf:
                    shutil.copy2(tmp_pdf, str(target.with_suffix('.pdf')))
                else:
                    # for non-pdf, copy original
                    shutil.copy2(str(f), str(out_dir / newname))

                # write txt summary
                txtpath = out_dir / (Path(newname).stem + '.txt')
                with open(txtpath, 'w', encoding='utf-8') as tf:
                    tf.write('INSTITUTION:\n' + str(inst) + '\n')
                    tf.write('OBJET:\n' + str(obj) + '\n')
                    tf.write('DATE:\n' + str(date) + '\n')
                    tf.write('\n--- VISION TEXT ---\n')
                    tf.write(vision_text or '')
                    tf.write('\n\n--- OCR TEXT ---\n')
                    tf.write(text_primary or '')

                summary.append((str(f), str(target), cert))
                print('  Reanalyzed ->', target.name, 'cert:', cert)
            except Exception as e:
                print('  Save failed:', e)

    print('\nDone. Summary:')
    for s in summary:
        print(' ', s)

if __name__ == '__main__':
    main()
