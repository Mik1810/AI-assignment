a = ['ok_sc_hg', 'alive_sc', 'ok_s1_ant']
b = ['ok_sc_hg', 'alive_sc', 'ok_s2_ant', 'ok_s2_trans']
c = ['ok_sc_lg', 'alive_sc', 'no_dist']
result = []
for i in range(len(a)):
    for j in range(len(b)):
        for k in range(len(c)):
            result.append((a[i], b[j], c[k]))

count = 1
new_result = []
for i in range(len(result)):
    new_result.append(frozenset(result[i]))
    print(f"{count}. {set(new_result[i])}")
    count += 1

print()
new_result = list(dict.fromkeys(new_result))

for i in range(len(new_result)):
    contatore = 0
    s = f"{i + 1}. "
    for item in new_result[i]:
        if item in a:
            contatore += 1
        if item in b:
            contatore += 1
        if item in c:
            contatore += 1
        s += item + ", "
    if contatore <= 3:
        print(s[:len(s)-2])
