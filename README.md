# 编程
* 数组
15. 三数之和
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

解析：
     
要求的是a+b+c=0 其实就是要求a+b=-c，a,b,c 中最小的数必为负数，我们可以设定c为其中最小的数。那么问题可以转化为依次遍历数组负元素c，然后在剩下的数中做两数之和为-c的问题。
     
问题在于如何简化算法以及优化复杂度。
     
1.首先可以先排序（O(nlogn)），这样保证数组有序之后可以利用大小关系判断。

2.由于 c最小，因此 a、b必在c 的右侧区域，在 c的右侧区域设置两个指针left、right，分别从其左边、右边向中间遍历； 一个 c 有可能有多个a,b满足条件，因此需要搜索完c右侧区域所有可能的情况；

3.去重，这一步则是利用了有序性，如果两个数相同，那他们在数组的位置一定是相邻的，因此去重的操作就能简单遍历一下相邻的是否相同即可。

```cpp
    vector<vector<int>> threeSum(vector<int>& nums) {
        int target;
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            if ((target = nums[i]) > 0) break;
            int l = i + 1, r = nums.size() - 1;
            while (l < r) {
                if (nums[l] + nums[r] + target < 0) ++l;
                else if (nums[l] + nums[r] + target > 0) --r;
                else {
                    ans.push_back({target, nums[l], nums[r]});
                    ++l, --r;
                    while(l < r && nums[l] == nums[l - 1]) ++l;
                    while(l < r && nums[r] == nums[r + 1]) --r;
                }
            }
        }
        return ans;
```

18. 四数之和

给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

注意：

答案中不可以包含重复的四元组。

示例：

给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]

解析：
转换为三数之和

```cpp
vector<vector<int>> fourSum(vector<int>& nums, int target) {
        if (nums.empty()) return {};
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        
        for (int z = 0; z < nums.size(); z++ ){
            if (z > 0 && nums[z] == nums[z-1]) continue;
            int newTarget = target - nums[z];

            for (int k = z + 1; k < nums.size(); k++) {
                if ( k > z+1 && nums[k] == nums[k-1]) continue;
                int newTarget2 = newTarget - nums[k];
                int l = k + 1, r = nums.size() - 1;
                while (l < r) {
                    if (nums[l] +  nums[r] < newTarget2) l ++;
                    else if (nums[l] +  nums[r] > newTarget2) r --;
                    else {
                        res.push_back({nums[z], nums[k], nums[l], nums[r]});
			++l, --r;
                        while (l < r && nums[l] == nums[l-1] ) l++;
                        while (l < r && nums[r] == nums[r+1] ) r--;
                    }
                }
            }
        }
        return res;
    }

```

46. 全排列

给定一个没有重复数字的序列，返回其所有可能的全排列。

示例:

输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

解析：
递归方式求解数组元素的全排列：
从s到e数组元素的全排列：
1. 若数组1个元素(s==e)，全排列就是它自己
2. 否则，遍历s到e数组元素, 第i趟全排列：i元素作为首元素 + s-1到e数组的全排列。

```cpp
vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        if (nums.size() == 0) {
           return ans; 
        }
        permutation(nums, 0, nums.size() - 1, ans);
        return ans;
    }
    void permutation(vector<int> nums, int s, int e, vector<vector<int>> &output) {
        //若数组1个元素，全排列就是它自己：
        if (s == e) {
            output.push_back(nums);
        }
        // s 到 e 的数组元素的全排列：
        // 遍历数组元素, 第i趟全排列：该元素作为首元素 + s-1到e数组的全排列。
        for (int i = s; i <= e; i++) {
           swap(nums[i], nums[s]);
           permutation(nums, s + 1, e, output);
           swap(nums[i], nums[s]);
        }
    }
```

31. 下一个排列

实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

解析：
题意是：找到给定数字列表的下一个字典排列（即找出这个数组排序出的所有数中，刚好比当前数大的那个数）

首先，对于任何降序排列，都没有可能的下一个更大的排列，题意说对于这种排列直接返回升序排列。

那么，非降序排列a[s,e]是我们主要处理的对象：

非降序排列中，如何找到比该排列更大的排列？

其应该是从右往左扫描找到首次破坏升序性的位置j，(即从右往左扫描，找到a[j] > a[j-1]的位置，此时a[j+1:e]是个升序排列) 

然后将j位置的元素和右侧升序序列第一个比它的元素交换，

再将 j 位置的右侧进行降序排列即可。

```cpp
void nextPermutation(vector<int>& nums) {
    if (nums.size() <= 1) return;
    int j = nums.size() - 1;
    while (j - 1 >= 0 && nums[j-1] >= nums[j]) j--;
    if (j == 0) {
        for (int i = 0; i < nums.size() / 2; ++i) {
            swap(nums[i], nums[nums.size() - 1 - i]);
        }
    } else {
        int r = nums.size() - 1;
        while (r >= j) {
            if (nums[r] > nums[j-1]) break;
            --r;
        }
        swap(nums[r], nums[j-1]);
        int r_mid_cnt =(nums.size() - j) /2;
        int i = 0;
        while (i < r_mid_cnt) {
            swap(nums[j+i], nums[nums.size() - 1 - i]);
            ++i;
        }
    }
    }
```

39. 组合总和

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 
示例 1:

输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]
示例 2:

输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]

解析：
回溯法，待解析...
```cpp
class Solution {

private:
    vector<int> candidates;
    vector<vector<int>> res;
    vector<int> path;
public:
    void DFS(int start, int target) {
            if (target == 0) {
                res.push_back(path);
                return;
            }
            for (int i = start;
                i < candidates.size(); i++) {
		if (target - candidates[i] < 0) break; // 该条路径无效，不再往下搜索。
                path.push_back(candidates[i]);
                DFS(i, target - candidates[i]);
                path.pop_back();
            }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        std::sort(candidates.begin(), candidates.end());
        this->candidates = candidates;
        DFS(0, target);

        return res;
    }
};
```

50. Pow(x, n)

实现 pow(x, n) ，即计算 x 的 n 次幂函数。

示例 1:

输入: 2.00000, 10
输出: 1024.00000
示例 2:

输入: 2.10000, 3
输出: 9.26100
示例 3:

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
说明:

-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

解析：
```cpp
double fastPow(double x, int n) {
        if (n == 0) {
            return 1.0;
        }
        double half = fastPow(x, n/2);
        if (n % 2 == 0) {
            return half * half;
        } else {
            return half * half * x;
        }
    }
    double myPow(double x, int n) {
        int abs_n = n >= 0 ? n:-n;
        double res = fastPow(x, n);
        return n >= 0? res:1/res;
    }
```

11. 盛最多水的容器

给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器，且 n 的值至少为 2。

示例:
   > 输入: [1,8,6,2,5,4,8,3,7]
   
   > 输出: 49

解析：
   基本的表达式: area = min(height[i], height[j]) * (j - i) 
   
   使用两个指针，值小的指针向内移动，这样就减小了搜索空间。
   
   因为面积取决于指针的距离与值小的值乘积，如果值大的值向内移动，距离一定减小，而求面积的另外一个乘数一定小于等于值小的值，因此面积一定减小，而我们要求最大的面积，因此值大的指针不动，而值小的指针向内移动遍历。
   
```cpp
    int maxArea(vector<int>& height) {
        if (height.size() <= 1) return -1;
        int ans = 0;
        int i = 0, j = height.size() - 1;
        while (i < j) {
            int h = min(height[i], height[j]);
            ans = max(ans, h * (j - i));
            if (height[i] < height[j]) ++i;
            else --j;
        }
        return ans;
    }
```
56. 合并区间

给出一个区间的集合，请合并所有重叠的区间。

示例 1:

输入: [[1,3],[2,6],[8,10],[15,18]]

输出: [[1,6],[8,10],[15,18]]

解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2:

输入: [[1,4],[4,5]]

输出: [[1,5]]

解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

解析：
```cpp
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.empty()) return {};
        sort(intervals.begin(),intervals.end());
        vector<vector<int>> ans;
        ans.push_back(intervals[0]);
        for (int i=1; i < intervals.size(); ++i) {
            if (ans.back()[1] >= intervals[i][0]) {
                ans.back()[1] = max(ans.back()[1], intervals[i][1]);
            } else {
                ans.push_back(intervals[i]);
            }
        }
        return ans;
    }
```

4. 寻找两个有序数组的中位数
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 nums1 和 nums2 不会同时为空。

示例 1:

nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
示例 2:

nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5

解析：
首先明确题意:
    1. 有序数组中位数，即中间位置的元素(数组奇数长度，中间位置即为中位数；数组偶长度，中间位置有两元素，这两元素平均即为数组的中位数)
    2. 需要时间复杂度为O(log (m+n))， 则需要用类似二分查找的分冶策略进行中位数查找。

这里我们需要定义一个函数来在两个有序数组中找到第K个元素，下面重点来看如何实现找到第K个元素。
首先，为了避免产生新的数组从而增加时间复杂度，我们使用两个变量i和j分别来标记数组nums1和nums2的起始位置。

然后来处理一些边界问题，比如当某一个数组的起始位置大于等于其数组长度时，说明其所有数字均已经被淘汰了，相当于一个空数组了，那么实际上就变成了在另一个数组中找数字，直接就可以找出来了。还有就是如果K=1的话，那么我们只要比较nums1和nums2的起始位置i和j上的数字就可以了。

难点就在于一般的情况怎么处理？因为我们需要在两个有序数组中找到第K个元素，为了加快搜索的速度，我们要使用二分法，对K二分，意思是我们需要分别在nums1和nums2中查找第K/2个元素，注意这里由于两个数组的长度不定，所以有可能某个数组没有第K/2个数字，所以我们需要先检查一下，数组中到底存不存在第K/2个数字，如果存在就取出来，否则就赋值上一个整型最大值。如果某个数组没有第K/2个数字，那么我们就淘汰另一个数字的前K/2个数字即可。有没有可能两个数组都不存在第K/2个数字呢，这道题里是不可能的，因为我们的K不是任意给的，而是给的m+n的中间值，所以必定至少会有一个数组是存在第K/2个数字的。最后就是二分法的核心啦，比较这两个数组的第K/2小的数字midVal1和midVal2的大小，如果第一个数组的第K/2个数字小的话，那么说明我们要找的数字肯定不在nums1中的前K/2个数字，所以我们可以将其淘汰，将nums1的起始位置向后移动K/2个，并且此时的K也自减去K/2，调用递归。反之，我们淘汰nums2中的前K/2个数字，并将nums2的起始位置向后移动K/2个，并且此时的K也自减去K/2，调用递归即可
```cpp
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        int left = (m + n + 1) / 2;
        int right = (m + n + 2) / 2;
        if (m == 0) {
            return (nums2[left - 1] + nums2[right - 1]) / 2.0;
        } else if (n == 0) {
            return (nums1[left - 1] + nums1[right - 1]) / 2.0;
        } else {
            return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0;
        }
    }
    
    int findKth(vector<int>& nums1, int i, vector<int>& nums2, int j, int k) {
        if (i >= nums1.size()) return nums2[j + k - 1];
        if (j >= nums2.size()) return nums1[i + k - 1];
        if (k == 1) {
            return min(nums1[i], nums2[j]);
        }
        int midVal1 = (i + k/2 - 1 < nums1.size())? nums1[i + k/2 - 1] : INT_MAX;
        int midVal2 = (j + k/2 - 1 < nums2.size())? nums2[j + k/2 - 1] : INT_MAX;
        if (midVal1 < midVal2) {
            return findKth(nums1, i + k/2, nums2, j, k - k/2);
        } else {
            return findKth(nums1, i, nums2, j + k/2, k - k/2);
        }
    }
```

 16. 最接近的三数之和
 
 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).

解析：
先排序, 然后遍历, 内部再使用双指针, 时间复杂度O(n²)
```cpp
int threeSumClosest(vector<int>& nums, int target) {
        if (nums.size() < 3) {
            return 0;
        }
        sort(nums.begin(), nums.end());
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i <  nums.size() - 2; ++i) {
            int l = i + 1;
            int r = nums.size() - 1;
            while (l < r) {
                int tmp_ans = nums[i] + nums[l] + nums[r];
                if (abs(tmp_ans - target) < abs(ans - target)) {
                    ans = tmp_ans;
                }
                if (tmp_ans < target) {
                    l++;
                } else if (tmp_ans > target) {
                    r--;
                } else {
                    return target;
                }
            }
        }
        return ans;
    }
```

75. 颜色分类

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意:

不能使用代码库中的排序函数来解决这道题。

示例:

输入: [2,0,2,1,1,0]

输出: [0,0,1,1,2,2]

进阶：

一个直观的解决方案是使用计数排序的两趟扫描算法。

首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。

你能想出一个仅使用常数空间的一趟扫描算法吗？

解析：

我们用三个指针（p0, p2 和curr）来分别追踪0的最右边界，2的最左边界和当前考虑的元素。

本解法的思路是沿着数组移动 curr 指针，若nums[curr] = 0，则将其与 nums[p0]互换；若 nums[curr] = 2 ，则与 nums[p2]互换。

```cpp
    void sortColors(vector<int>& nums) {
        // 对于所有 idx < p0 : nums[idx < p0] = 0
        // curr 是当前考虑元素的下标
        int p0 = 0, curr = 0;
        // 对于所有 idx > p2 : nums[idx > p2] = 2
        int p2 = nums.size() - 1;

        while (curr <= p2) {
        if (nums[curr] == 0) {
            swap(nums[curr++], nums[p0++]);
        }
        else if (nums[curr] == 2) {
            swap(nums[curr], nums[p2--]);
        }
        else curr++;
        }
    }
```

148. 排序链表

归并排序。

由于题目要求空间复杂度是 O(1)，因此不能使用递归。因此这里使用 bottom-to-up 的算法来解决。

bottom-to-up 的归并思路是这样的：先两个两个的 merge，完成一趟后，再 4 个4个的 merge，直到结束。举个简单的例子：[4,3,1,7,8,9,2,11,5,6].

step=1: (3->4)->(1->7)->(8->9)->(2->11)->(5->6)
step=2: (1->3->4->7)->(2->8->9->11)->(5->6)
step=4: (1->2->3->4->7->8->9->11)->5->6
step=8: (1->2->3->4->5->6->7->8->9->11)

链表里操作最难掌握的应该就是各种断链啊，然后再挂接啊。在这里，我们主要用到链表操作的两个技术：

* merge(l1, l2)，双路归并。
* cut(l, n)，可能有些同学没有听说过，它其实就是一种 split 操作，即断链操作。不过我感觉使用 cut 更准确一些，它表示，将链表 l 切掉前 n 个节点，并返回后半部分的链表头。
* 额外再补充一个 dummyHead 大法。

掌握了这三大神器后，我们的 bottom-to-up 算法伪代码就十分清晰了：
```cpp
current = dummy.next;
tail = dummy;
for (step = 1; step < length; step *= 2) {
	while (current) {
		// left->@->@->@->@->@->@->null
		left = current;

		// left->@->@->null   right->@->@->@->@->null
		right = cut(current, step); // 将 current 切掉前 step 个头切下来。

		// left->@->@->null   right->@->@->null   current->@->@->null
		current = cut(right, step); // 将 right 切掉前 step 个头切下来。
		
		// dummy.next -> @->@->@->@->null，最后一个节点是 tail，始终记录
		//                        ^
		//                        tail
		tail.next = merge(left, right);
		while (tail->next) tail = tail->next; // 保持 tail 为尾部
	}
}
```
比较正式的代码。

```cpp
 ListNode* sortList(ListNode* head) {
        ListNode dummyHead(0);
        dummyHead.next = head;
        auto p = head;
        int length = 0;
        while (p) {
            ++length;
            p = p->next;
        }
        
        for (int size = 1; size < length; size <<= 1) {
            auto cur = dummyHead.next;
            auto tail = &dummyHead;
            
            while (cur) {
                auto left = cur;
                auto right = cut(left, size); // left->@->@ right->@->@->@...
                cur = cut(right, size); // left->@->@ right->@->@  cur->@->...
                
                tail->next = merge(left, right);
                while (tail->next) {
                    tail = tail->next;
                }
            }
        }
        return dummyHead.next;
    }
    
    ListNode* cut(ListNode* head, int n) {
        auto p = head;
        while (--n && p) {
            p = p->next;
        }
        
        if (!p) return nullptr;
        
        auto next = p->next;
        p->next = nullptr;
        return next;
    }
    
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode dummyHead(0);
        auto p = &dummyHead;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                p->next = l1;
                p = l1;
                l1 = l1->next;       
            } else {
                p->next = l2;
                p = l2;
                l2 = l2->next;
            }
        }
        p->next = l1 ? l1 : l2;
        return dummyHead.next;
    }
```

* 动态规划

53. 最大子序和

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

示例:

输入: [-2,1,-3,4,-1,2,1,-5,4],

输出: 6

解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

解析：
扫描数组元素S，每个时刻 t 都计算max_continues_sum_t：以t时刻为结束点的连续子数组(S[t-k,...,t])的最大和。
数组扫描完，会有N 个max_continues_sum_t，其中最大的即为整个数组的最大子序和。
```cpp
    int maxSubArray(vector<int>& nums) {
        if(nums.size() == 0) return 0;
        int res = nums[0];
        int last_max_continuos_sum = 0; // 记录以当前时刻t为结束点的连续子数组(s[t-k,...,t])的最大和。 
        for (int i = 0; i < nums.size(); ++i) {
            if (last_max_continuos_sum > 0) { //s[i-k, .. i-1] 的子数组和为正，和k时刻元素连接起来，会有正向收益，可作为k时刻的连续子数组。
                last_max_continuos_sum += nums[i];
            } else { //s[i-k, .. i-1] 的子数组和为负，若和k时刻元素连接起来，会有负向收益，因此 k 时刻不和前面的数组相连接。
                last_max_continuos_sum = nums[i];
            }
            res = max(res, last_max_continuos_sum);
        }
        return res;
    }
```

79. 单词搜索

给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

示例:

board =
[

  ['A','B','C','E'],
  
  ['S','F','C','S'],
  
  ['A','D','E','E']
  
]

给定 word = "ABCCED", 返回 true.

给定 word = "SEE", 返回 true.

给定 word = "ABCB", 返回 false.

解析：
```cpp
 bool exist(vector<vector<char>>& board, string word) {
        if (board.size() <=0) {
            return false;
        }
        int n = board.size();
        int m = board[0].size();
        //记录目前已被占用的网格点(防止一个网格点被重复使用)
        vector<vector<bool>> masked(n, vector<bool>(m, false));
        //每个网格点都可以朝四个方向搜索：下，左，上，右； 方向顺序无要求
        vector<vector<int>> directions = {{0, 1}, {-1, 0}, {0, -1}, {1, 0}}; 
        //对board每个位置都尝试搜索，看能不能找到 word
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (search_word(board, i, j, word, 0, masked, directions) == true) {
                    return true;
                }
            }
        }
        return false;
    }
    //从board 的(i,j)位置，按direction限制的方向搜索 word[index:-1]
    bool search_word(vector<vector<char>>& board, int i, int j, string & word, int index, vector<vector<bool>>& masked, vector<vector<int>>& directions) {
        //递归停止条件
        if (word.size() - 1 == index) {
            return board[i][j] == word[index];
        }
        int n = board.size();
        int m = board[0].size();
        // （i,j）网格点匹配了word[index]，再继续往下搜索
        if (board[i][j] == word[index]) {
            // 先mask（i,j）, 若最终搜不到，再还原
            masked[i][j] = true;
            //深度优先，四个方向都可尝试搜索
            for (vector<int> direction : directions) {
                int try_i = i + direction[0] ;
                int tyr_j = j + direction[1];
                //尝试的下一个位置(try_i,tyr_j)合法，且当前未被占用, 若从该位置开始能搜到 word[index+1:-1]（递归search_word），则返回 true；否则，继续尝试其他方向搜索。
                if (try_i < n && try_i >=0 &&
                    tyr_j < m && tyr_j >=0 &&
                    !masked[try_i][tyr_j] &&
                    search_word(board, try_i, tyr_j, word, index + 1, masked, directions))
                    return true;
            }
            masked[i][j] = false;
        }
        return false;
    }
```

5. 最长回文子串

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例 1：

输入: "babad"

输出: "bab"

注意: "aba" 也是一个有效答案。

示例 2：

输入: "cbbd"

输出: "bb"

解析:

```cpp
string longestPalindrome(string s) {
        if (s.empty()) return "";
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.size(); i++) {
            //尝试以 i 为中心进行扩展
            int len1 = expandAroundCenter(s, i, i);
            //尝试以 i,i+1中间位置为中心进行扩展
            int len2 = expandAroundCenter(s, i, i + 1);
            //取最长的扩展
            int len = max(len1, len2);
            if (len > right - left) {
                left = i - (len -1) / 2;
                right = i + len / 2;
            }
        }
        //cout<<left<<endl;
        //cout<<right<<endl;
        return s.substr(left, right - left + 1);
    }
    int expandAroundCenter(string & s, int left, int right) {
        if (left > right && right < s.size()) return 0;
        int l = left;
        int r = right;
        while (l >=0 && r < s.size() && s[l] == s[r]) {
            --l;
            ++r;
        }
        //实际字符串长度。r, l为实际位置的下一个位置, 故实际长度为：r - l + 1 -2
        return r - l - 1;
    }
```

70. 爬楼梯

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 n 是一个正整数。

示例 1：

输入： 2

输出： 2

解释： 有两种方法可以爬到楼顶。

1.  1 阶 + 1 阶

2.  2 阶

示例 2：

输入： 3

输出： 3

解释： 有三种方法可以爬到楼顶。

1.  1 阶 + 1 阶 + 1 阶

2.  1 阶 + 2 阶

3.  2 阶 + 1 阶

解析:

```cpp
//记忆化递归 
//时间复杂度：O(n)，树形递归的大小可以达到 n。
//空间复杂度：O(n)，递归树的深度可以达到 n。
    int climbStairs(int n) {
        if (n < 0) {
            return 0;
        }
        int memo[n + 1] = {0};
        return climb(0, n, memo);
    }
    int climb(int i, int n, int memo[]) {
        if (i > n) {
            return 0;
        }
        if (i == n) {
            return 1;
        }
        if (memo[i] > 0) {
            return memo[i];
        }
        memo[i] = climb(i + 1, n, memo) + climb(i + 2, n, memo);
        return memo[i];
    }
    
//动态规划
//时间复杂度：O(n)，单循环到 n。
//空间复杂度：O(n)，dp数组用了n的空间。
/*
第 i阶可以由以下两种方法得到：
在第 (i-1)阶后向上爬1阶。
在第 (i-2) 阶后向上爬 2 阶。
所以到达第 i阶的方法总数就是到第 (i-1)阶和第 (i-2)阶的方法数之和。
令 dp[i]表示能到达第 i阶的方法总数：
dp[i]=dp[i-1]+dp[i-2]
*/
public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
```

300. 最长上升子序列

给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:

输入: [10,9,2,5,3,7,101,18]

输出: 4 

解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。

说明:

可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。

你算法的时间复杂度应该为 O(n2) 。

解析：

```cpp
 int lengthOfLIS(vector<int>& nums) {
        if (nums.size() <= 0) return 0;
        //dp[i]:第i个字符作为结束字符的最长上升子序列长度
        vector<int> dp(nums.size(), 1);
        for (int i = 1; i < nums.size(); ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
```


* 二叉搜索树

二叉查找树（英语：Binary Search Tree），也称为 二叉搜索树、有序二叉树（Ordered Binary Tree）或排序二叉树（Sorted Binary Tree），是指一棵空树或者具有下列性质的二叉树：

- 节点的左子树仅包含键小于节点键的节点。

- 节点的右子树仅包含键大于节点键的节点。

- 左右子树也必须是二叉搜索树。


*树

105. 从前序与中序遍历序列构造二叉树

根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

前序遍历 preorder = [3,9,20,15,7]

中序遍历 inorder = [9,3,15,20,7]

返回如下的二叉树：

>    3
>   / \
>  9  20
>    /  \
>   15   7

解析：
首先要知道一个结论，前序/后序+中序序列可以唯一确定一棵二叉树，所以自然而然可以用来建树。

看一下前序和中序有什么特点，前序1,2,4,7,3,5,6,8 ，中序4,7,2,1,5,3,8,6；

有如下特征：

前序：[根结点，左子树结点，右子树结点]；

中序：[左子树结点，根结点，右子树结点]，

前序中左起第一位是根结点root，我们可以据此找到中序中根结点的位置rootin；继而可得出左子树结点个数为rootin；

可递归构造此树：

```cpp
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if (preorder.size() == 0 || inorder.size() == 0) {
            return nullptr;
        }
        int root_val = preorder[0];
        TreeNode* p_root = new TreeNode(root_val);
        vector<int>::iterator it = find(inorder.begin(), inorder.end(), root_val);
        int r_dis_in = distance(inorder.begin(), it);
        
        vector<int> l_preorder(preorder.begin() + 1, preorder.begin() + 1 + r_dis_in);
        vector<int> l_inorder(inorder.begin(), inorder.begin() + r_dis_in);
        p_root->left = buildTree(l_preorder, l_inorder);

        vector<int> r_preorder(preorder.begin() + 1 + r_dis_in, preorder.end());
        vector<int> r_inorder(inorder.begin() + r_dis_in + 1, inorder.end());
        p_root->right = buildTree(r_preorder, r_inorder);
        
        return p_root;
    }
```

141. 环形链表

给定一个链表，判断链表中是否有环。

解析：使用快慢指针检测是否有换

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if (head == nullptr) {
            return false;
        }
        ListNode * slow = head;
        ListNode * fast = head->next;
        while (fast && fast->next) {
            if (slow == fast) {
                return true;
            }
            fast = fast->next->next;
            slow = slow->next;
        }
        return false;
    }
};
```

 * 排序
 
 147. 对链表进行插入排序
 
 插入排序的动画演示如上。从第一个元素开始，该链表可以被认为已经部分排序（用黑色表示）。
每次迭代时，从输入数据中移除一个元素（用红色表示），并原地将其插入到已排好序的链表中。

 

插入排序算法：

插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。
 

示例 1：

输入: 4->2->1->3
输出: 1->2->3->4
示例 2：

输入: -1->5->3->4->0
输出: -1->0->3->4->5

解析：

想要排序块，就要尽可能少的做比较

需要一个指针指向当前已排序的最后一个位置，这里用的是head指针

需要另外一个指针pre,每次从表头循环，这里用的是dummy表头指针。

每次拿出未排序的节点，先和前驱比较，如果大于或者等于前驱，就不用排序了，直接进入下一次循环

如果前驱小，则进入内层循环，依次和pre指针比较，插入对应位置即可。

```cpp
    ListNode* insertionSortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* dummy = new ListNode(0);
        dummy->next  = head;
        ListNode* pre = nullptr;
        while (head != nullptr && head->next != nullptr) {
	    //head 是有序链表的最后一个元素
            if (head->val <= head->next->val) {
                head = head->next;
                continue;
            }
            pre = dummy;
            while (pre->next->val < head->next->val) {
                pre = pre->next;
            }

            //断链，接链
            ListNode* curr = head->next;
            head->next = curr->next;
            curr->next = pre->next;
            pre->next = curr;
        }
        return dummy->next;
    }
```

179. 最大数

给定一组非负整数，重新排列它们的顺序使之组成一个最大的整数。

示例 1:

输入: [10,2]

输出: 210

示例 2:

输入: [3,30,34,5,9]

输出: 9534330

说明: 输出结果可能非常大，所以你需要返回一个字符串而不是整数。

解析：
数组中元素e1,e2, 若 e2e1 > e1e2，则e2应在e1之前，才能组成最终的大数；对数组中所有元素，按上述原则进行排序，最终便得到了最大数。

```cpp
    string largestNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), [](int& a, int &b) {
            string sa = to_string(a);
            string sb = to_string(b);
            return sa + sb > sb + sa;
        });
        string res;
        bool nonzero = false;
        for (int e : nums) {
            res += to_string(e);
            if (e) nonzero = true;
        }
        return nonzero ? res : "0";
    }
```

* 数学

2. 两数相加

给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例：

输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)

输出：7 -> 0 -> 8

原因：342 + 465 = 807

解析:
和平常手算两数相加一样，从低位往高位逐位相加，用 carray 缓存当前的进位。
```cpp
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* p = l1;
        ListNode* q = l2;
        ListNode* res_head = new ListNode(0);
        ListNode* prev = res_head;
        int carry = 0;
        while(p != nullptr || q != nullptr || carry != 0) {
            int p_val = (p != nullptr)? p->val:0;
            int q_val = (q != nullptr)? q->val:0;
            int sum = carry + p_val + q_val;
            carry = sum / 10;
            ListNode* cur_res = new ListNode(sum % 10);
            prev->next = cur_res;
            prev = prev->next;
            p = (p != nullptr)? p->next: nullptr;
            q = (q != nullptr)? q->next: nullptr;
        }
        return res_head->next;
    }
```

7. 整数反转

给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

示例 1:

输入: 123

输出: 321

示例 2:

输入: -123

输出: -321

示例 3:

输入: 120

输出: 21

注意:

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

解析:
我们想重复“弹出”x的最后一位数字，并将它“推入”到 rev的后面。最后，rev 将与x相反。要在没有辅助堆栈 / 数组的帮助下 “弹出” 和 “推入” 数字，我们可以使用数学方法。
```cpp
    int reverse(int x) {
        int rev = 0;
        int pop = 0;
        while (x != 0) {
            pop = x % 10;
            x = x / 10;

            if (rev > INT_MAX / 10 || ((rev == (INT_MAX / 10)) && pop > 7)) return 0;
            if (rev < INT_MIN / 10 || ((rev == (INT_MIN / 10)) && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }
```

60. 第k个排列

给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

说明：

给定 n 的范围是 [1, 9]。
给定 k 的范围是[1,  n!]。
示例 1:

输入: n = 3, k = 3
输出: "213"

示例 2:

输入: n = 4, k = 9
输出: "2314"

解析：
```cpp
    string getPermutation(int n, int k) {
         /**
        可以用数学的方法来解, 因为数字都是从1开始的连续自然数, 排列出现的次序可以推
        算出来, 对于n=4, k=15 找到k=15排列的过程:

        确定第一位:
            k = 14(从0开始计数)
            index = k / (n-1)! = 2, 说明第15个数的第一位是3 
            更新k
            k = k - index*(n-1)! = 2
        确定第二位:
            k = 2
            index = k / (n-2)! = 1, 说明第15个数的第二位是2
            更新k
            k = k - index*(n-2)! = 0
        确定第三位:
            k = 0
            index = k / (n-3)! = 0, 说明第15个数的第三位是1
            更新k
            k = k - index*(n-3)! = 0
        确定第四位:
            k = 0
            index = k / (n-4)! = 0, 说明第15个数的第四位是4
        最终确定n=4时第15个数为3214 
        **/
        string res = "";
        vector<int> candidates;
        //分母的阶乘数
        int factorials[n + 1] = {0};
        factorials[0] = 1;
        int fact = 1;
        for (int i = 1; i <= n; ++i) {
            candidates.push_back(i);
            fact *= i;
            factorials[i] = fact;
        }
        k -= 1;
        int index = 0;
        for (int i= n-1; i >= 0; --i) {
            // 计算候选数字的index
            index = k / factorials[i];
            res += to_string(candidates[index]);
            candidates.erase(candidates.begin() + index);
            k -= index * factorials[i];
        }
        return res;
    }
```

* 回溯算法


