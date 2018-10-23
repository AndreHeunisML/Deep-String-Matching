# Format positive examples for abbreviation data

import pandas as pd


def get_all_abb():
    abbs1 = {"abridged": "abr",
             "abstract": "abstr",
             "academy": "acad",
             "adaptation": "adapt",
             "American": "Amer",
             "annotation": "annot",
             "annual": "annu",
             "association": "assoc",
             "augmented": "augm",
             "authorized": "authoriz",
             "biannual": "biannu",
             "bibliography": "bibliogr",
             "bimonthly": "bimonth",
             "biography": "biogr",
             "brochure": "broch",
             "bulletin": "bull",
             "catalog": "cat",
             "centimeter": "cm",
             "chapter": "chap",
             "commission": "commiss",
             "company": "co",
             "compiler": "comp",
             "conference": "conf",
             "column": "col",
             "corporation": "corp",
             "department": "dept",
             "diagram": "diagr",
             "dictionary": "dict",
             "director": "dir",
             "directory": "dir",
             "dissertation": "diss",
             "distribution": "dist",
             "division": "div",
             "Doctor": "Dr",
             "document": "doc",
             "edition": "ed",
             "editor": "ed",
             "encyclopedia": "encycl",
             "English": "Engl",
             "enlarged": "enl",
             "European": "Europ",
             "executive": "exec",
             "explanation": "expl",
             "extract": "extr",
             "facsimile": "facs",
             "faculty": "ac",
             "figure": "fig",
             "foundation": "found",
             "frontispiece": "front",
             "gazette": "gaz",
             "government": "gov",
             "handbook": "handb",
             "illustration": "ill",
             "illustrator": "ill",
             "impression": "impr",
             "inch": "in",
             "inclusive": "incl",
             "incomplete": "incompl",
             "index": "ind",
             "information": "inform",
             "institute": "inst",
             "international": "intern",
             "introduction": "introd",
             "invariable": "invar",
             "laboratory": "lab",
             "library": "libr",
             "literature": "lit",
             "manual": "man",
             "manuscript": "ms",
             "meeting": "meet",
             "microfiche": "mfiche",
             "microfilm": "mf",
             "millimeter": "mm",
             "miscellaneous": "misc",
             "modified": "mod",
             "monograph": "monogr",
             "monthly": "month",
             "national": "nat",
             "new series": "n.s",
             "newspaper": "newsp",
             "notice": "not",
             "number": "no",
             "observation": "observ",
             "original": "orig",
             "pamphlet": "pamph",
             "paperback": "pbk",
             "part": "pt",
             "periodical": "period",
             "photography": "phot",
             "picture": "pict",
             "portrait": "portr",
             "posthumous": "posth",
             "preface": "pref",
             "preliminary": "prelim",
             "preparation": "prep",
             "preprint": "prepr",
             "printed": 'print',
             "proceedings": "proc",
             "professor": "prof",
             "program": "progr",
             "pseudonym": "pseud",
             "publication": "publ",
             "publisher": "publ",
             "quarterly": "quart",
             "reference": "ref",
             "reprint": "repr",
             "reproduction": "reprod",
             "responsible": "resp",
             "revised": "rev",
             "scientific": "sci",
             "section": "sect",
             "separate": "sep",
             "series": "ser",
             "session": "sess",
             "society": "soc",
             "special": "spec",
             "successor": "success",
             "summary": "summ",
             "supplement": "suppl",
             "symposium": "symp",
             "table": "tab",
             "translation": "transl",
             "translator": "transl",
             "transliteration": "translit",
             "university": "univ",
             "volume": "vol",
             "year": "y",
             "yearbook": "yb"}

    anchor = []
    pos = []

    # Use abbreviations as anchor
    for key, val in abbs1.items():
        anchor.append(key)
        pos.append(val)
    abbs1 = pd.DataFrame(data={'anchor': anchor, 'pos_match': pos})

    abbs2 = pd.read_csv("data/abbv/abbreviations.csv")
    abbs2 = abbs2.iloc[7:][["Abbreviation", "Meaning (English)"]]
    abbs2.columns = ["anchor", "pos_match"]

    return pd.concat([abbs1, abbs2])


if __name__ == "__main__":
    result = get_all_abb()
    print(len(result))
    print(result)