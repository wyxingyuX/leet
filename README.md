## 编程
  * 数组
    * 三数之和
     [tab][tab]给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
     [tab][tab]注意：答案中不可以包含重复的三元组。
     [tab][tab]解析：
     
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
   *  盛最多水的容器
   [tab][tab] 给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
   [tab][tab] 说明：你不能倾斜容器，且 n 的值至少为 2。
   [tab][tab] 示例:
   > 输入: [1,8,6,2,5,4,8,3,7]
   
   > 输出: 49
   
   [tab][tab] 解析：
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
   * 合并区间
   [tab][tab] 给出一个区间的集合，请合并所有重叠的区间。

   [tab][tab]  示例 1:

   [tab][tab]  输入: [[1,3],[2,6],[8,10],[15,18]]
   [tab][tab]  输出: [[1,6],[8,10],[15,18]]
   [tab][tab]  解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
   [tab][tab]  示例 2:

   [tab][tab]  输入: [[1,4],[4,5]]
   [tab][tab]  输出: [[1,5]]
   [tab][tab]  解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
   
   [tab][tab] 解析：
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

* 4. 寻找两个有序数组的中位数
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


   
   

