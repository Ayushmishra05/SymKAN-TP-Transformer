import re
import pandas as pd



input_file = 'QED_data/QED_data.txt'
output_file = 'QED_data/processed_dataset.csv'



pattern = r'Interaction:(.*?)\s*:\s*.*:\s*(.*?)\s*:\s*(.*)'

data = []
current_block = ""



with open(input_file, 'r') as file:
    for line in file:
        if 'Interaction:' in line and current_block:
            match = re.search(pattern, current_block, re.DOTALL)
            if match:
                input_text = match[1].strip()
                squared_amplitude = match[3].strip()

                input_text = re.sub(r'\bInteraction\b', '', input_text)
                input_text = re.sub(r'\bOffShell\b', '', input_text)
                input_text = re.sub(r'\bVertex\b', '', input_text)

                input_text = re.sub(r'e_\w+_\d+', 'e_[ID]', input_text)
                input_text = re.sub(r'p_\d+', 'p_[ID]', input_text)
                input_text = re.sub(r'X_\d+', 'X_[ID]', input_text)


                input_text = re.sub(r'gamma_{%.*?}', 'gamma_{...}', input_text)

                input_text = re.sub(r'\w+_{\w+}', '[TENSOR]', input_text)

                input_text = re.sub(r'\((.*?)\)', '(X)', input_text)   # Keep (X) for position terms
                input_text = re.sub(r'\^\((.*?)\)', '^(*)', input_text) # Keep ^(*) for conjugates

    
                input_text = re.sub(r'u_\(\*\)', 'u_(*)', input_text)
                input_text = re.sub(r'v_\(\*\)', 'v_(*)', input_text)


                input_text = re.sub(r'\s+', ' ', input_text).strip()


                data.append({
                    'text': input_text,
                    'label': squared_amplitude
                })

            current_block = ""

        current_block += line

    if current_block:
        match = re.search(pattern, current_block, re.DOTALL)
        if match:
            input_text = match[1].strip()
            squared_amplitude = match[3].strip()

            input_text = re.sub(r'\bInteraction\b', '', input_text)
            input_text = re.sub(r'\bOffShell\b', '', input_text)
            input_text = re.sub(r'\bVertex\b', '', input_text)

            input_text = re.sub(r'e_\w+_\d+', 'e_[ID]', input_text)
            input_text = re.sub(r'p_\d+', 'p_[ID]', input_text)
            input_text = re.sub(r'X_\d+', 'X_[ID]', input_text)
            input_text = re.sub(r'gamma_{%.*?}', 'gamma_{...}', input_text)
            input_text = re.sub(r'\w+_{\w+}', '[TENSOR]', input_text)

            input_text = re.sub(r'\((.*?)\)', '(X)', input_text)
            input_text = re.sub(r'\^\((.*?)\)', '^(*)', input_text)

            input_text = re.sub(r'u_\(\*\)', 'u_(*)', input_text)
            input_text = re.sub(r'v_\(\*\)', 'v_(*)', input_text)
            input_text = re.sub(r'\s+', ' ', input_text).strip()

            data.append({
                'text': input_text,
                'label': squared_amplitude
            })


df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(df.head())
