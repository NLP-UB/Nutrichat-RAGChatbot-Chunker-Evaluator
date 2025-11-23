import pandas as pd

def recursive_split_to_excel_verbose_force_sep(text, separators, chunk_size=50, excel_file="chunks_verbose.xlsx"):
    records = []
    chunks = []

    def recursive_split(text, separators, depth=0):
        if not separators:
            records.append({
                "depth": depth,
                "separator": None,
                "part_index": 0,
                "part_text": text.strip(),
                "part_len": len(text.strip()),
                "action": "accept"
            })
            return [text.strip()]

        sep = separators[0]
        sep_verbose = None
        if sep == "\n":
            sep_verbose = "\\n"
        elif sep == "\n\n":
            sep_verbose = "\\n\\n"
        elif sep == " ":
            sep_verbose = "' '"
        else:
            sep_verbose = sep if sep else None

        # Catat langkah sebelum split
        records.append({
            "depth": depth,
            "separator": sep_verbose,
            "part_index": None,
            "part_text": text.strip(),
            "part_len": len(text.strip()),
            "action": "split attempt"
        })

        # Split text jika ada separator
        if sep and sep in text:
            parts = text.split(sep)
        else:
            parts = [text]

        result = []
        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) > chunk_size and len(separators) > 1:
                records.append({
                    "depth": depth,
                    "separator": sep_verbose,
                    "part_index": i,
                    "part_text": part,
                    "part_len": len(part),
                    "action": "recurse"
                })
                result.extend(recursive_split(part, separators[1:], depth=depth+1))
            else:
                records.append({
                    "depth": depth,
                    "separator": sep_verbose,
                    "part_index": i,
                    "part_text": part,
                    "part_len": len(part),
                    "action": "accept"
                })
                result.append(part)
        return result

    all_parts = recursive_split(text, separators)

    # Merge ke chunk berdasarkan chunk_size
    temp = ""
    for part in all_parts:
        if len(temp) + len(part) + 1 <= chunk_size:
            temp += part + " "
            records.append({
                "depth": "chunking",
                "separator": "' '",
                "part_index": None,
                "part_text": part,
                "part_len": len(part),
                "action": f"merge into temp: '{temp.strip()}'"
            })
        else:
            if temp:
                chunks.append(temp.strip())
                records.append({
                    "depth": "chunking",
                    "separator": None,
                    "part_index": None,
                    "part_text": temp.strip(),
                    "part_len": len(temp.strip()),
                    "action": "finalize chunk"
                })
            temp = part
            records.append({
                "depth": "chunking",
                "separator": None,
                "part_index": None,
                "part_text": temp.strip(),
                "part_len": len(temp.strip()),
                "action": "start new temp chunk"
            })

    if temp:
        chunks.append(temp.strip())
        records.append({
            "depth": "chunking",
            "separator": None,
            "part_index": None,
            "part_text": temp.strip(),
            "part_len": len(temp.strip()),
            "action": "finalize last chunk"
        })

    # Simpan ke Excel
    df = pd.DataFrame(records)
    df.to_excel(excel_file, index=False)
    print(f"[INFO] Saved verbose recursive split to '{excel_file}' with {len(chunks)} chunks.")

    return chunks

# Contoh penggunaan
text = ("Bronkopneumonia pada anak usia dini sering kali disebabkan oleh infeksi bakteri, virus, atau jamur "
        "yang menyerang saluran napas bagian bawah. Gejala yang umum meliputi demam tinggi, batuk berdahak, "
        "sesak napas, dan napas cepat. Pada kasus berat, dapat terjadi penurunan kesadaran dan kegagalan pernapasan. "
        "Penanganan medis yang tepat sangat penting untuk mencegah komplikasi lebih lanjut. Selain itu, "
        "pemberian asuhan gizi yang sesuai juga berperan dalam proses pemulihan pasien.")
separators = ["\n\n", "\n", ".", " ", ""]
chunk_size = 50

chunks = recursive_split_to_excel_verbose_force_sep(text, separators, chunk_size=chunk_size, excel_file="chunks_verbose.xlsx")
