s = open("route3_mve.py").read()
old = "if len(S) > 1200 or len(U) > 400: return None"
new = "if len(S) > 1200 or len(U) > 600: return None"
assert s.count(old) == 1, "count=%d" % s.count(old)
open("route3_mve.py", "w").write(s.replace(old, new))
print("patched route3_mve.py")
