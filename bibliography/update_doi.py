import bibtexparser
import requests

def fetch_doi(entry):
    title   = entry.get('title','')
    authors = entry.get('author','').split(' and ')
    query   = f"{title} {authors[0] if authors else ''}"
    r = requests.get(
        'https://api.crossref.org/works',
        params={'query.bibliographic': query,'rows':1}
    )
    items = r.json().get('message',{}).get('items',[])
    return items[0].get('DOI') if items else None

# 1) Leia o .bib original
with open('masslump-bib.bib', encoding='utf-8') as f:
    bib = bibtexparser.load(f)

updated = False
# 2) Para cada entrada sem DOI, tenta buscar
for e in bib.entries:
    if 'doi' not in e:
        doi = fetch_doi(e)
        if doi:
            e['doi'] = doi
            updated = True
            print(f"✔ {e['ID']}: {doi}")

# 3) Salva versão atualizada
if updated:
    with open('masslump-bib-updated.bib','w',encoding='utf-8') as f:
        bibtexparser.dump(bib,f)
    print("\nArquivo salvo: masslump-bib-updated.bib")
else:
    print("Nenhuma entrada faltando DOI.")
