from pathlib import Path

my_file = Path("sta.jpg")
if my_file.is_file():
    print("yes it is")
