# Data Structure & Algoritms

## Basic

## String

```python
s='String'
```

## List
```python
l=list()
```

## Dictionary
```python
d=dict()
```

## Other Primitive Data types

## Linked List
```python
class Node:
    def __init__(self,val,next=None):
        self.val=val
        self.next=next

class LinkedList:
    def __init__(self,val):
        self.head=Node(val)
```

## Tree
```python
class Node:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

class LinkedList:
    def __init__(self,val):
        self.head=Node(val)
```

## Graph

## Traversing
```python
l=[1,2,3,4,5]
n=len(l)
for i in range(n):
    print(l[i])
```

## Searching
```python
l=[1,2,3,4,5]
n=len(l)
se= 4
for i in range(n):
    if l[i]==se:
        print(i)
print(-1)
```

## Sorting

Sorting is a technique to sort any list/array either in increasing or decreasing order. There are multiple sorting techniques.

### Bubble Sort
- **Mechanism** : Repeatedly swap adjacent elements if they are in the wrong order.
- **Complexity**: `O(n^2)` for average and worst-case, O(n) for best case (already sorted).
- **Key Point**: Simple but inefficient.

```python
def bubbleSort(ar,n):
    for i in range(n):
        swapped = False
        for j in range(n-1-i):
            if ar[j]>ar[j+1]:
                ar[j],ar[j+1]=ar[j+1], ar[j]
                swapped = True
        if not swapped:
            return

import random
n = random.randrange(10,15)
ar = [random.randrange(10, 100) for _ in range(n)]
print(f"List before sorting {ar}")
bubbleSort(ar,n)
print(f"List after sorting {ar}")
```

### Selection Sort

- **Mechanism**: Divide the array into a sorted and an unsorted region. Repeatedly pick the smallest (or largest) element from the unsorted region and add it to the sorted region.
- **Complexity**: O(n^2) for average, worst, and best cases.
- **Key Point**: Inefficient but simple.
```python
def selectionSort(ar,n):
    for i in range(n):
        min_ind=i
        for j in range(i+1,n):
            if ar[j]<ar[min_ind]:
                min_ind=j
        ar[min_ind],ar[i]=ar[i],ar[min_ind]

import random
n = random.randrange(10,15)
ar = [random.randrange(10, 100) for _ in range(n)]
print(f"List before sorting {ar}")
selectionSort(ar,n)
print(f"List after sorting {ar}")
```

### Insertion Sort

- **Mechanism**: Build the final sorted array one item at a time by repeatedly removing one element from the input and inserting it into its correct position within the sorted list.
- **Complexity**: O(n^2) for average and worst-case, O(n) for best case (already sorted).
- **Key Point**: Efficient for small lists or nearly sorted lists.
```python
def insertionSort(ar,n):
    for i in range(1,n):
        min_ele=ar[i]
        j=i-1
        while j>=0 and min_ele<ar[j]:
            ar[j+1]=ar[j]
            j-=1
        ar[j+1]=min_ele


import random
n = random.randrange(10,15)
ar = [random.randrange(10, 100) for _ in range(n)]
print(f"List before sorting {ar}")
insertionSort(ar,n)
print(f"List after sorting {ar}")
```

### Merge Sort

- **Mechanism**: Divide the array in half, sort each half, and then merge them back together.
- **Complexity**: O(n log n) for average, worst, and best cases.
- **Key Point**: Divide and conquer approach. Stable sort. Requires O(n) additional space.

```python
def merge(ar,low,mid,high):
    left=ar[low:mid+1]
    right=ar[mid+1:high+1]
    k=low
    i=j=0
    while i<len(left) and j<len(right):
        if left[i]<right[j]:
            ar[k]=left[i]
            i+=1
        else:
            ar[k]=right[j]
            j+=1
        k+=1
    
    while i<len(left):
        ar[k]=left[i]
        i+=1
        k+=1
    
    while j<len(right):
        ar[k]=right[j]
        j+=1
        k+=1

def mergeSort(ar,low, high):
    if low<high:
        mid=(low+high)//2
        mergeSort(ar,low,mid)
        mergeSort(ar,mid+1,high)
        merge(ar,low,mid,high)



import random
n = random.randrange(10,15)
ar = [random.randrange(10, 100) for _ in range(n)]
print(f"List before sorting {ar}")
mergeSort(ar,0,n-1)
print(f"List after sorting {ar}")
```






## Back Tracking

## Dynamic Programming

## Subarray
A subarray is a contiguous or non-empty portion of an array. In the context of an array, a subarray is a subset of the original array that maintains the relative order of the elements.

### Maximum Subarray
Given an integer array nums, find the subarray with the largest sum, and return its sum.

```python
# Return sum of maximum subarray
def sumMaxSubarray(ar):
    n=len(ar)
    if n<1:
        return 0
    c_sum = m_sum = ar[0]
    for x in ar[1:]:
        if x>c_sum+x:
            c_sum=x
        else:
            c_sum+=x
        if c_sum>m_sum:
            m_sum=c_sum
    return m_sum

# Example usage:
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum = sumMaxSubarray(nums)
print("Maximum subarray sum:", max_sum)
```

Given an integer array nums, find the subarray with the largest sum, and return the subarray.

```python
# Return the maximum subarray
def maxSubarray(ar):
    n=len(ar)
    if n<1:
        return 0
    c_sum = m_sum = ar[0]
    c_start=0
    st=end=0
    for i in range(n):
        x=ar[i]
        if x>c_sum+x:
            c_sum=x
            c_start = i
        else:
            c_sum+=x
        if c_sum>m_sum:
            m_sum=c_sum
            st=c_start
            end=i
    return ar[st:end+1]

# Example usage:
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_subarray = maxSubarray(nums)
print("Maximum subarray :", max_subarray)
```

