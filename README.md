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
   
   

