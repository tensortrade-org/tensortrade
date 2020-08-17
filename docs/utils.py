


with open("source/examples/renderers_and_plotly_chart.md") as search:

    code_block = False
    for line in search:
        if line.startswith("```"):
            code_block = not code_block

        if not code_block and line.startswith("##"):
            text = line.replace("#", "").strip()
            print(f"<br>**{text}**<br>", type(line))
