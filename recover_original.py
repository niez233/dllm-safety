# recover_original.py
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="从带有 evaluation_result 的文件还原为原始格式（去掉该字段），写入新文件。")
    parser.add_argument("input_json", type=str, help="被覆盖的JSON文件路径（含 evaluation_result）")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出文件路径（默认：原文件名加 _original.json）")
    args = parser.parse_args()

    in_path = args.input_json
    out_path = args.output or in_path.replace(".json", "_original.json")

    # 读取
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容：文件可以是列表或单个对象
    if isinstance(data, dict):
        items = [data]
        single_object = True
    else:
        items = data
        single_object = False

    # 去掉 evaluation_result 字段
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            cleaned.append(item)  # 非字典条目，原样保留
            continue
        new_item = dict(item)    # 浅拷贝
        new_item.pop("evaluation_result", None)  # 删除多余字段
        cleaned.append(new_item)

    # 如果原始是单对象，则还原为单对象；否则保持列表
    result = cleaned[0] if (single_object and cleaned) else cleaned

    # 写入新文件
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 已写入：{out_path}")

if __name__ == "__main__":
    main()