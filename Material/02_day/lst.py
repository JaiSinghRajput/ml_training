lst = ["hello",34,67,45,78,True,5.6]
print(type(lst))
print(len(lst))
print(lst.count(34))
print(lst.index(34))
lst.append("blue")
lst.extend(["red","hello" ,"green"])
lst.insert(1, "yellow")
print(lst)
lst.sort(reverse=True,)
print(lst)
print(lst[::-1])
lst.pop()
lst.pop(3) #remove value of 3 index
print(lst)
lst.remove("hello") # remove first accurance of hello in lst
print(lst)
lst.clear()