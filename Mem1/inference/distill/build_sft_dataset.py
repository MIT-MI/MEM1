import json
import argparse

def convert(input_paths, output_path):
    all_conversations = []

    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(json.loads(line))

        for item in data:
            if '1' in input_path:
                if item["Exact_match"] != 1:
                    continue
            elif '2' in input_path:
                if item["Exact_match"] != 2:
                    continue
            elif '3' in input_path:
                if item["Exact_match"] != 3:
                    continue

            # determine how many steps (max j where t_j and r_j exist)
            max_j = -1
            j = 0
            while f"t{j}" in item and f"r{j}" in item:
                max_j = j
                j += 1

            # for each step j, build one user-assistant pair
            for j in range(max_j+1):
                # build user content
                if j == 0:
                    user_content = item["q"].strip()
                else:
                    parts = [item["q"].strip()]
                    parts.append(item[f"t{j-1}"].strip())
                    parts.append(item[f"r{j-1}"].strip())

                    all_conversations.append({
                        "messages": [
                            {"role": "user",      "content": "\n".join(parts[:2])},
                            {"role": "assistant", "content": parts[2]}
                        ]
                    })

                    # include information if present
                    if f"i{j-1}" in item:
                        parts.append(item[f"i{j-1}"].strip())
                    user_content = "\n".join(parts)

                # build assistant content: t_j then r_j
                t_j = item[f"t{j}"].strip()
                r_j = item[f"r{j}"].strip()
                assistant_content = "\n".join([t_j, r_j])

                # append this pair as its own messages object
                all_conversations.append({
                    "messages": [
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })
    print(len(all_conversations))

    # write out
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_files = ["nqsearch_multi_2_train_reconstruction_dicts_gpt-4o_mem1.jsonl", "nqsearch_multi_3_train_reconstruction_dicts_gpt-4o_mem1.jsonl", "nqsearch_1_train_reconstruction_dicts_gpt-4o_mem1.jsonl"]
    output_file = "RAG_train_sft.json"

    convert(input_files, output_file)