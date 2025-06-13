n = int(input("Enter the number of elements in the list: "))
arr = []
for i in range(n):
    num = int(input(f"Enter element {i + 1}:  "))
    arr.append(num)

start = 0
end = n - 1
while start < end: 
    arr[start], arr[end] = arr[end], arr[start]
    start += 1
    end -= 1
print("The reversed list is:", arr)
    