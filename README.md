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
     class Solution {
public:
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
    }
};
     ```

