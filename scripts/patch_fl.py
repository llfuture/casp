s = open("fl_mve.py").read()
old = '"d": {str(j): {str(i): c[j][i] for i in range(m)} for j in range(n)}}'
new = '"d": {str(i): {str(j): c[j][i] for j in range(n)} for i in range(m)}}'
assert s.count(old) == 1, "old not found once: %d" % s.count(old)
open("fl_mve.py", "w").write(s.replace(old, new))
print("patched fl_mve.py")
