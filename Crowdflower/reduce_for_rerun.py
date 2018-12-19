import csv
def read_current_annotations(filename):
    counts={}
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            counts[row["id"]]=counts.get(row["id"],0)+1

        return counts


def rewrite_set(filename,counts):
    f = open("new11_"+filename,"w",newline='',encoding="utf-8")
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)

        writer = csv.DictWriter(f,fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["check_one_gold"]!="11":
                continue
            if "ID" in row:
                id = row["ID"]
            else:
                id = row["id"]
            count = counts.get(id,0)

            if count < 5:
                writer.writerow(row)
    f.close()



stats = read_current_annotations("ident_current.csv")
rewrite_set("comparison.csv",stats)

stats = read_current_annotations("sentence_current.csv")
rewrite_set("sentence.csv",stats)