def obfuscate_code(input_code):
    # Replace variable names
    obfuscated_code = input_code.replace('original_variable', 'obfuscated_variable')
    return obfuscated_code


if __name__ == '__main__':
    input_code = 'original_variable = 123'
    result = obfuscate_code(input_code)
    print(result)