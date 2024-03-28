class FixLabels:
    def __init__(self):
        self.file_path = None
    def fix_labels(self,file_path):
        with open(file_path, 'r') as f:
            content = f.readlines()
            recontent = []
            for i in range(0, len(content)):
                s = list(content[i])
                if s[0] != "0":
                    s[0] = "0"
                s = "".join(s)
                recontent.append(s)
            f.close()
        with open(file_path, 'w') as f:
            f.writelines(recontent)
            f.close()
        return "Complete"