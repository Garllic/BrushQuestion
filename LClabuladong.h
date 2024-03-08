#pragma once
#include "tool.h"

using namespace std;

class LClabuladong
{
public:
    LClabuladong() {}
    //208. 实现 Trie (前缀树)
    template<typename T1>
    class MyTrie
    {
    private:
        class TrieNode
        {
        public:
            bool exist;
            T1 val;
            TrieNode* children[58];
            vector<int> index_of_childrens;
        };
        //返回指向word最后一个字符结点的指针
        static TrieNode* getTrieNode(TrieNode* root, string word) {
            TrieNode* p = root;
            int i = 0;
            for (i = 0; i < word.size(); i++) {
                char temp_char = word[i];
                if (p->children[temp_char - 'A'] == NULL) {
                    return NULL;
                }
                p = p->children[temp_char - 'A'];
            }
            return p;
        }
        //返回从一个节点开始存在的所有键
        static void getAllKeysWithNode(TrieNode* p, vector<string>& res, string word) {
            if (p->exist) {
                res.push_back(word);
            }
            for (int i : p->index_of_childrens) {
                char temp_char = i + 'A';
                getAllKeysWithNode(p->children[i], res, word + temp_char);
            }
            return;
        }
        //返回一个前缀的键的值sum
        static void getSumWithNode(TrieNode* p, int& res) {
            if (p->exist) {
                res += p->val;
            }
            for (int i : p->index_of_childrens) {
                getSumWithNode(p->children[i], res);
            }
            return;
        }
    public:
        TrieNode* root;
        TrieNode* cur;
        int size;

        MyTrie() {
            root = new TrieNode();
            size = 0;
            cur = root;
        }

        int getSize() {
            return this->size();
        }

        void insert(string word, T1 val = 0) {
            TrieNode* p = this->root;
            for (int i = 0; i < word.size(); i++) {
                char temp_char = word[i];
                if (p->children[temp_char - 'A'] != NULL) {
                    p = p->children[temp_char - 'A'];
                    continue;
                }
                TrieNode* temp_node = new TrieNode();
                p->children[temp_char - 'A'] = temp_node;
                p->index_of_childrens.push_back(temp_char - 'A');
                p = p->children[temp_char - 'A'];
            }
            if (p->exist == false) {
                p->exist = true;
                this->size++;
            }
            p->val = val;
        }

        void remove(string word) {

        }

        bool hasKey(string word) {
            TrieNode* p = getTrieNode(root, word);
            if (!p) {
                return false;
            }
            return p->exist;
        }


        //("themxyz")->them，最短前缀是them
        string shortestPrefixOf(string query) {
            string res;
            TrieNode* p = root;
            for (int i = 0; i < query.size(); i++) {
                char temp_char = query[i];
                if (p->children[temp_char - 'A'] != NULL) {
                    p = p->children[temp_char - 'A'];
                    res += temp_char;
                    if (p->exist) {
                        return res;
                    }
                    continue;
                }
                if (p->exist) {
                    return res;
                }
                else {
                    return "";
                }
            }
        }

        //("themxyz")->them，最长前缀是them
        string longestPrefixOf(string query) {

        }

        //前缀为prefix的所有键
        vector<string> keysWithPrefix(string prefix) {
            vector<string> res;
            TrieNode* p = getTrieNode(this->root, prefix);
            if (p == NULL) {
                return res;
            }
            getAllKeysWithNode(p, res, prefix);
            return res;
        }

        //前缀为prefix的键是否存在
        bool hasKeyWithPrefix(string prefix) {
            TrieNode* p = getTrieNode(this->root, prefix);
            return p != NULL;
        }

        //通配符'.'，搜索匹配的所有键
        vector<string> keysWithPattern(string pattern) {

        }

        //通配符'.'，是否存在匹配的键
        bool hasKeyWithPattern(string pattern) {
            TrieNode* p = root;
            for (int i = 0; i < pattern.size(); i++) {
                char temp_char = pattern[i];
                if (temp_char == '.') {
                    for (int j : p->index_of_childrens) {
                        char temp = j;
                        pattern[i] = temp;
                        TrieNode* temp_root = root;
                        root = p;
                        bool flag = hasKeyWithPattern(pattern.substr(i, -1));
                        root = temp_root;
                        if (flag) {
                            return true;
                        }
                    }
                    return false;
                }
                if (p->children[temp_char - 'A'] == NULL) {
                    return false;
                }
                p = p->children[temp_char - 'A'];
            }
            return p->exist;
        }

        T1 getKeyVal(string word) {
            TrieNode* p = getTrieNode(root, word);
            if (p == NULL) {
                return 0;
            }
            if (!p->exist) {
                return 0;
            }
            return p->val;
        }

        int sum(string prefix) {
            int n;
            int res = 0;
            TrieNode* p = getTrieNode(this->root, prefix);
            if (p == NULL) {
                return 0;
            }
            getSumWithNode(p, res);

            return res;
        }
    };
    //355. 设计推特
    class Twitter
    {
    private:
        class User;
        class PersonTwitter;

        class User {
        public:
            int user_id;
            list<User*> followed;
            PersonTwitter* head;

            User() {}
            User(int id);

            void post(PersonTwitter* twee);
            void follow(User* followee);
            void unfollow(User* unfollowee);
            vector<int> getNewTwitter();
        };

        class PersonTwitter {
        public:
            int id;
            int time;
            PersonTwitter* next;

            PersonTwitter() {}
            PersonTwitter(int i, int timestamp);
        };
    public:
        int timestamp;
        unordered_map<int, User*> users;

        Twitter();

        void postTweet(int userId, int tweetId);
        vector<int> getNewsFeed(int userId);
        void follow(int followerId, int followeeId);
        void unfollow(int followerId, int followeeId);
    };
    //46. 全排列
    vector<vector<int>> res;
    void fullArrangement(list<int>& nums, vector<int>& path);
    vector<vector<int>> permute(vector<int>& nums);
    //322. 零钱兑换
    int* coinChange_memo;
    int coinChange_dp(vector<int>& coins, int amount);
    int coinChange(vector<int>& coins, int amount);
    //300. 最长递增子序列
    int* lengthOfLIS_memo;
    int lengthOfLIS(vector<int>& nums);
    //354. 俄罗斯套娃信封问题
    static bool maxEnvelopes_cmp(const vector<int>& v1, const vector<int>& v2);
    int maxEnvelopes(vector<vector<int>>& envelopes);
    //931. 下降路径最小和
    int minFallingPathSum(vector<vector<int>>& matrix);
    //72. 编辑距离
    vector<vector<int>> minDistance_memo;
    int minDistance_min(int a, int b, int c);
    int minDistance_dp(string word1, int i, string word2, int j);
    int minDistance(string word1, string word2);
    int minDistance2(string word1, string word2);
    //53. 最大子数组和
    int maxSubArray_dp(vector<int>& nums);
    int maxSubArray_SlideWindow(vector<int>& nums);
    int maxSubArray_PrefixSum(vector<int>& nums);
    //1143. 最长公共子序列
    int longestCommonSubsequence(string text1, string text2);
    //583. 两个字符串的删除操作
    int minDeleteDistance(string word1, string word2);
    //712. 两个字符串的最小ASCII删除和
    int minimumDeleteSum(string s1, string s2);
    //1312. 让字符串成为回文串的最少插入次数
    int minInsertions(string s);
    //516. 最长回文子序列
    int longestPalindromeSubseq(string s);
    //416. 分割等和子集
    bool canPartition(vector<int>& nums);
    //12. 整数转罗马数字
    void addRomanNum(int num, char arr_roman, string& res);
    string intToRoman(int num);
    //49. 字母异位词分组
    vector<vector<string>> groupAnagrams(vector<string>& strs);
    //62. 不同路径
    int uniquePaths(int m, int n);
    //131. 分割回文串
    string s_val;
    vector<vector<string>> partitionPalindromic_res;
    void startPartition(vector<vector<bool>>& memo, int begin, vector<string>& strs);
    vector<vector<string>> partitionPalindromic(string s);
    //494. 目标和
    int findTargetSumWays(vector<int>& nums, int target);
    //279. 完全平方数
    int numSquares(int n);
    //377. 组合总和 IV
    int combinationSum4(vector<int>& nums, int target);
    //518. 零钱兑换 II
    int coinChangeII(int amount, vector<int>& coins);
    //64. 最小路径和
    int minPathSum(vector<vector<int>>& grid);
    //174. 地下城游戏
    int calculateMinimumHP(vector<vector<int>>& dungeon);
    //514. 自由之路
    unordered_map<char, list<int>> findRotateSteps_char_to_index;
    vector<vector<int>> findRotateSteps_memo;
    int findRotateSteps(string ring, string key);
    int findRotateSteps_dp(string ring, int i, string key, int j);
    //17. 电话号码的字母组合
    vector<string> letterCombinations_ret;
    unordered_map<int, string> letterCombinations_digit_to_letter;
    vector<string> letterCombinations(string digits);
    void digits2letters(string digits, int begin, string res);
    //24. 两两交换链表中的节点
    ListNode* swapPairs(ListNode* head);
    //99. 恢复二叉搜索树
    vector<int> recoverTree_res;
    void recoverTree(TreeNode* root);
    void recoverTree_dfs(TreeNode* root, vector<TreeNode*>& order);
    //51. N 皇后
    vector<vector<string>> solveNQueens_res;
    vector<vector<string>> solveNQueens(int n);
    void solveNQueens_recursion(vector<string>& chessboard, int queen_order);
    bool solveNQueens_isAvailable(vector<string>& chessboard, int row, int col);
    //698. 划分为k个相等的子集
    bool canPartitionKSubsets(vector<int>& nums, int k);
    bool canPartitionKSubsets_loadBuckets(vector<int>& buckets, int bucket_index, vector<int>& nums, int nums_index);
    //200. 岛屿数量
    int numIslands(vector<vector<char>>& grid);
    void numIslands_connect(vector<vector<char>>& grid, int i, int j);
    //1254. 统计封闭岛屿的数目
    int closedIsland(vector<vector<int>>& grid);
    void closedIsland_isClosedIsland(vector<vector<int>>& grid, int i, int j);
    void closedIsland_reverse(vector<vector<int>>& grid, int i, int j);
    //1020. 飞地的数量
    int numEnclaves(vector<vector<int>>& grid);
    void numEnclaves_isClosed(vector<vector<int>>& grid, int i, int j, int& ret);
    void numEnclaves_reverse(vector<vector<int>>& grid, int i, int j);
    //695. 岛屿的最大面积
    int maxAreaOfIsland(vector<vector<int>>& grid);
    int maxAreaOfIsland_countArea(vector<vector<int>>& grid, int i, int j);
    //1905. 统计子岛屿
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2);
    void countSubIslands_islandReverse(vector<vector<int>>& grid, int i, int j, int pave, int val);
    //111. 二叉树的最小深度
    int minDepth(TreeNode* root);
    //752. 打开转盘锁
    int openLock(vector<string>& deadends, string target);
    //773. 滑动谜题
    int slidingPuzzle(vector<vector<int>>& board);
    bool slidingPuzzle_isOK(vector<vector<int>>& board);
    void slidingPuzzle_swapTwoSquare(vector<vector<int>>& board, int i1, int j1, int i2, int j2);
    pair<int, int> slidingPuzzle_findZero(vector<vector<int>>& board);
    bool slidingPuzzle_isVisited(vector<vector<int>>& board, bool visited[6][6][6][6][6][6]);
    void slidingPuzzle_visit(vector<vector<int>>& board, bool visited[6][6][6][6][6][6]);
    //787. K 站中转内最便宜的航班
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k);
    vector<vector<int>> findCheapestPrice_creatPriceTable(vector<vector<int>>& flights, int n);
    //486. 预测赢家
    bool PredictTheWinner(vector<int>& nums);
    //299. 猜数字游戏
    string getHint(string secret, string guess);
    //238. 除自身以外数组的乘积
    vector<int> productExceptSelf(vector<int>& nums);
    //1922. 统计好数字的数目
    int countGoodNumbers(long long n);
    //139. 单词拆分
    bool wordBreak(string s, vector<string>& wordDict);
    //43. 字符串相乘
    string multiply(string num1, string num2);
    vector<int> multiply_add(vector<int> num1, vector<int> num2);
    //400. 第 N 位数字
    int findNthDigit(int n);
    //390. 消除游戏
    int lastRemaining(int n);
    //789. 逃脱阻碍者
    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target);
    int escapeGhosts_distance(vector<int>& source, vector<int>& target);
    //873. 最长的斐波那契子序列的长度
    int lenLongestFibSubseq(vector<int>& arr);
    //1262. 可被三整除的最大和
    int maxSumDivThree(vector<int>& nums);
    //688. 骑士在棋盘上的概率
    double knightProbability(int n, int k, int row, int column);
    double knightProbability_getProbability(int n, int x, int y, vector<vector<double>>& dp);
    //967. 连续差相同的数字
    vector<int> numsSameConsecDiff(int n, int k);
    void numsSameConsecDiff_recursion(int n, int k, int p, vector<int>& cur);
    vector<int> numsSameConsecDiff_ret;
    //10. 正则表达式匹配
    bool isMatch(string s, string p);
    //887. 鸡蛋掉落
    unordered_map<int, int> superEggDrop_memo;
    int superEggDrop(int k, int n);
    int superEggDrop_dp(int k, int n);
    //877. 石子游戏
    bool stoneGame(vector<int>& piles);
    //198. 打家劫舍
    int rob(vector<int>& nums);
    //机器调度
    int MachineHandling(int n, vector<int> d1, vector<int> d2);
    void MachineHandling_judgement(int t1, int t2, int& temp1, int& temp2);
    //871. 最低加油次数
    int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations);
    //213. 打家劫舍 II
    int robII(vector<int>& nums);
    //337. 打家劫舍 III
    int robIII(TreeNode* root);
    int robIII_cursion(TreeNode* root);
    //2560. 打家劫舍 IV
    int minCapability(vector<int>& nums, int k);
    bool minCapability_check(vector<int>& nums, int k, int m);
    //121. 买卖股票的最佳时机
    int maxProfit(vector<int>& prices);
    //122. 买卖股票的最佳时机 II
    int maxProfitII(vector<int>& prices);
    //123. 买卖股票的最佳时机 III
    int maxProfitIII(vector<int>& prices);
    //66. 加一
    vector<int> plusOne(vector<int>& digits);
    //188. 买卖股票的最佳时机 IV
    int maxProfitIV(int k, vector<int>& prices);
    //309. 最佳买卖股票时机含冷冻期
    int maxProfitFreezing(vector<int>& prices);
    //714. 买卖股票的最佳时机含手续费
    int maxProfitV(vector<int>& prices, int fee);
    //8. 找出字符串中第一个匹配项的下标
    int strStr(string haystack, string needle);
    int strStr_getIndex(string& needle, vector<int>& kmp_arr, int cur);
    //435. 无重叠区间
    int eraseOverlapIntervals(vector<vector<int>>& intervals);
    //1024. 视频拼接
    int videoStitching(vector<vector<int>>& clips, int time);
    //45. 跳跃游戏 II
    int jump(vector<int>& nums);
    //55. 跳跃游戏
    bool canJump(vector<int>& nums);
    //52. N 皇后 II
    int totalNQueens(int n);
    void totalNQueens_recursion(int row);
    bool totalNQueens_mapIsOk(int row, int col);
    int totalNQueens_ret;
    vector<vector<char>> totalNQueens_map;
    //216. 组合总和 III
    vector<vector<int>> combinationSum3(int k, int n);
    void combinationSum3_recursion(int k, int n, vector<vector<int>>& buckets, int& v_sum, vector<int>& cur, int x);
    //40. 组合总和 II
    vector<vector<int>> combinationSum2_ret;
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
    void combinationSum2_backtrace(vector<int>& candidates, int target, vector<int>& cur, int sum, int x);
    //47. 全排列 II
    vector<vector<int>> permuteUnique_ret;
    vector<vector<int>> permuteUnique(vector<int>& nums);
    void permuteUnique_backtrace(vector<int>& nums, vector<int>& cur, vector<bool>& nums_flags, int x);
    //90. 子集 II
    vector<vector<int>> subsetsWithDup_ret;
    vector<vector<int>> subsetsWithDup(vector<int>& nums);
    void subsetsWithDup_backtrace(vector<int>& nums, vector<int>& cur, int x);
    //37. 解数独
    vector<vector<char>> solveSudoku_ret;
    bool solveSudoku_IsOk;
    void solveSudoku(vector<vector<char>>& board);
    void solveSuduku_InitGrid(vector<vector<char>>& board, vector<vector<bool>>& grid_flag);
    void solveSuduku_BackTrace(vector<vector<char>>& board, vector<vector<bool>>& grid_flag, int grid_num, int number);
    bool solveSuduku_RemedyIsOk(vector<vector<char>>& board, int x, int y, int value);
    //22. 括号生成
    vector<string> generateParenthesis_ret;
    vector<string> generateParenthesis(int n);
    void generateParenthesis_BackTrace(int n, int index, string& cur, int left_num, int right_num);
    //382. 链表随机节点
    ListNode* getRandom_head;
    LClabuladong(ListNode* head);
    int getRandom();
    //384. 打乱数组
    vector<int> shuffle_arr;
    LClabuladong(vector<int>& nums);
    vector<int> reset();
    vector<int> shuffle();
    //398. 随机数索引
    int RandomPick(int target);
    //136. 只出现一次的数字
    int singleNumber(vector<int>& nums);
    //191. 位1的个数
    int hammingWeight(uint32_t n);
    //231. 2 的幂
    bool isPowerOfTwo(int n);
    //268. 丢失的数字
    int missingNumber(vector<int>& nums);
    //172. 阶乘后的零
    long long trailingZeroes(long long n);
    //793. 阶乘函数后 K 个零
    int preimageSizeFZF(int k);
    long long preimageSizeFZF_LeftBound(int k);
    long long preimageSizeFZF_rightBound(int k);
    //204. 计数质数
    int countPrimes(int n);
    //372. 超级次方
    int superPow_mod = 1337;
    int superPow(int a, vector<int>& b);
    int pow1(int x, int n);
    //645. 错误的集合
    vector<int> findErrorNums(vector<int>& nums);
    //292. Nim 游戏
    bool canWinNim(int n);
    //319. 灯泡开关
    int bulbSwitch(int n);
    //241. 为运算表达式设计优先级
    vector<int> diffWaysToCompute_ret;
    vector<int> diffWaysToCompute(string expression);
    vector<int> diffWaysToCompute_GetComupute(vector<string> expression, int begin, int end);
    vector<string> diffWaysToCompute_GetOpVector(string expression);
    //1288. 删除被覆盖区间
    int removeCoveredIntervals(vector<vector<int>>& intervals);
    //56. 合并区间
    vector<vector<int>> merge(vector<vector<int>>& intervals);
    //986. 区间列表的交集
    vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList);
    //659. 分割数组为连续子序列
    bool isPossible(vector<int>& nums);
    //969. 煎饼排序
    vector<int> pancakeSort(vector<int>& arr);
    //224. 基本计算器
    int calculate(string s);
    //42. 接雨水
    int trap(vector<int>& height);
    int trap_recalculation(vector<int> height, int startIndex, int endIndex);
    //391. 完美矩形
    bool isRectangleCover(vector<vector<int>>& rectangles);
    //855. 考场就座
    class ExamRoom {
    private:
        int nextIndex;
        vector<bool> seats;
        vector<int> diffOfSeats;
        int num;
    public:
        ExamRoom(int n);
        int seat();
        void leave(int p);
    };
    //392. 判断子序列
    bool isSubsequence(string s, string t);
    //792. 匹配子序列的单词数
    int numMatchingSubseq(string s, vector<string>& words);
    int numMatchingSubseq_FindNext(vector<int>& charRecord, int count);
    //977. 有序数组的平方
    vector<int> sortedSquares(vector<int>& nums);
    //367. 有效的完全平方数
    bool isPerfectSquare(int num);
    //844. 比较含退格的字符串
    bool backspaceCompare(string s, string t);
    //904. 水果成篮
    int totalFruit(vector<int>& fruits);
    //59. 螺旋矩阵 II
    vector<vector<int>> generateMatrix(int n);
    void generateMatrix_Padding(vector<vector<int>>& matrix, int xStart, int yStart, int value, int n);
    //54. 螺旋矩阵
    vector<int> spiralOrder(vector<vector<int>>& matrix);
    //203. 移除链表元素
    ListNode* removeElements(ListNode* head, int val);
    //707. 设计链表
    class MyLinkedList {
    private:
        struct Node
        {
            int val;
            Node* next;
            Node* prev;
            Node(int n) :val(n), next(nullptr), prev(nullptr) {};
        };
        Node* head;
        int size;
    public:
        MyLinkedList();
        int get(int index);
        void addAtHead(int val);
        void addAtTail(int val);
        void addAtIndex(int index, int val);
        void deleteAtIndex(int index);
    };
    //242. 有效的字母异位词
    bool isAnagram(string s, string t);
    //383. 赎金信
    bool canConstruct(string ransomNote, string magazine);
    //349. 两个数组的交集
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2);
    //350. 两个数组的交集 II
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2);
    //202. 快乐数
    bool isHappy(int n);
    //1. 两数之和
    vector<int> twoSum(vector<int>& nums, int target);
    //454. 四数相加 II
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4);
    //15. 三数之和
    vector<vector<int>> threeSum(vector<int>& nums);
    //18. 四数之和
    vector<vector<int>> fourSum(vector<int>& nums, int target);
    //344. 反转字符串
    void reverseString(vector<char>& s);
    //541. 反转字符串 II
    string reverseStr(string s, int k);
    //151. 反转字符串中的单词
    string reverseWords(string s);
    vector<string> reverseWords_GetWords(string s);
    //459. 重复的子字符串
    bool repeatedSubstringPattern(string s);
    //1047. 删除字符串中的所有相邻重复项
    string removeDuplicates(string s);
    //150. 逆波兰表达式求值
    int evalRPN(vector<string>& tokens);
    int evalRPN_Compute(int a, int b, char op);
    //347. 前 K 个高频元素
    vector<int> topKFrequent(vector<int>& nums, int k);
    //145. 二叉树的后序遍历
    vector<int> postorderTraversal_ret;
    vector<int> postorderTraversal(TreeNode* root);
    void postorderTraversal_recursion(TreeNode* root);
    //102. 二叉树的层序遍历
    vector<vector<int>> levelOrder(TreeNode* root);
    //589. N 叉树的前序遍历
    vector<int> preorder(Node* root);
    //590. N 叉树的后序遍历
    vector<int> postorder(Node* root);
    //101. 对称二叉树
    bool isSymmetric(TreeNode* root);
    bool isSymmetric_Compare(TreeNode* left, TreeNode* right);
    //100. 相同的树
    bool isSameTree(TreeNode* p, TreeNode* q);
    //572. 另一棵树的子树
    bool isSubtree(TreeNode* root, TreeNode* subRoot);
    bool isSubtree_Contain(TreeNode* root, TreeNode* subRoot);
    //110. 平衡二叉树
    bool isBalanced(TreeNode* root);
    int isBalanced_GetDiff(TreeNode* root);
    //257. 二叉树的所有路径
    vector<string> binaryTreePaths(TreeNode* root);
    vector<string> binaryTreePaths_GetPath(TreeNode* root);
    //404. 左叶子之和
    int sumOfLeftLeaves_ret;
    int sumOfLeftLeaves(TreeNode* root);
    void sumOfLeftLeaves_Recursion(TreeNode* root);
    //513. 找树左下角的值
    int findBottomLeftValue_h;
    int findBottomLeftValue_ret;
    int findBottomLeftValue(TreeNode* root);
    void findBottomLeftValue_Recursion(TreeNode* root, int h);
    //112. 路径总和
    bool hasPathSum(TreeNode* root, int targetSum);
    bool hasPathSum_Recursion(TreeNode* root, int targetSum);
    //530. 二叉搜索树的最小绝对差
    int getMinimumDifference(TreeNode* root);
    int getMinimumDifference_Recursion(TreeNode* root);
    int getMinimumDifference_GetMax(TreeNode* root);
    int getMinimumDifference_GetMin(TreeNode* root);
    //501. 二叉搜索树中的众数
    vector<int> findMode(TreeNode* root);
    void findMode_TreeToVector(TreeNode* root, vector<int>& nodeList);
    //108. 将有序数组转换为二叉搜索树
    TreeNode* sortedArrayToBST(vector<int>& nums);
    TreeNode* sortedArrayToBST_Traverse(vector<int>& nums);
    //93. 复原 IP 地址
    vector<string> restoreIpAddresses_ret;
    vector<string> restoreIpAddresses(string s);
    void restoreIpAddresses_Traverse(string s, string cur, int n);
    //491. 递增子序列
    vector<vector<int>> findSubsequences_ret;
    vector<vector<int>> findSubsequences(vector<int>& nums);
    void findSubsequences_Traverse(vector<int>& nums, int begin, vector<int> cur);
    //332. 重新安排行程
    unordered_map<string, map<string, int>> findItinerary_targets;
    vector<string> findItinerary(vector<vector<string>>& tickets);
    bool findItinerary_Traverse(int ticketNum, vector<string>& result);
    //455. 分发饼干
    int findContentChildren(vector<int>& g, vector<int>& s);
    //376. 摆动序列
    int wiggleMaxLength(vector<int>& nums);
    //1005. K 次取反后最大化的数组和
    int largestSumAfterKNegations(vector<int>& nums, int k);
    //134. 加油站
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost);
    //135. 分发糖果
    int candy(vector<int>& ratings);
    //860. 柠檬水找零
    bool lemonadeChange(vector<int>& bills);
    //406. 根据身高重建队列
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people);
    //452. 用最少数量的箭引爆气球
    int findMinArrowShots(vector<vector<int>>& points);
    //763. 划分字母区间
    vector<int> partitionLabels(string s);
    //738. 单调递增的数字
    int monotoneIncreasingDigits(int n);
    //968. 监控二叉树
    int minCameraCover_ret;
    int minCameraCover(TreeNode* root);
    int minCameraCover_recursion(TreeNode* cur);
    //746. 使用最小花费爬楼梯
    int minCostClimbingStairs(vector<int>& cost);
    //63. 不同路径 II
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
    //343. 整数拆分
    int integerBreak(int n);
    //1049. 最后一块石头的重量 II
    int lastStoneWeightII(vector<int>& stones);
    //474. 一和零
    int findMaxForm(vector<string>& strs, int m, int n);
    pair<int, int> findMaxForm_CountZeroOne(string s);
    //674. 最长连续递增序列
    int findLengthOfLCIS(vector<int>& nums);
    //718. 最长重复子数组
    int findLength(vector<int>& nums1, vector<int>& nums2);
    //1035. 不相交的线
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2);
    //115. 不同的子序列
    int numDistinct(string s, string t);
    //647. 回文子串
    int countSubstrings(string s);
    //84. 柱状图中最大的矩形
    int largestRectangleArea(vector<int>& heights);
    //2605. 从两个数字数组里生成最小数字
    int minNumber(vector<int>& nums1, vector<int>& nums2);
    //1123. 最深叶节点的最近公共祖先
    TreeNode* lcaDeepestLeaves(TreeNode* root);
    void lcaDeepestLeaves_CountDeep(TreeNode* root, int d, map<int, vector<TreeNode*>>& deepNodes);
    int lcaDeepestLeaves_FindParents(TreeNode* root, int n, TreeNode*& res);
    //2594. 修车的最少时间
    long long repairCars(vector<int>& ranks, int cars);
    bool repairCars_CanRepair(vector<int>& ranks, int cars, long long time);
    //2651. 计算列车到站时间
    int findDelayedArrivalTime(int arrivalTime, int delayedTime);
    //1222. 可以攻击国王的皇后
    vector<vector<int>> queensAttacktheKing(vector<vector<int>>& queens, vector<int>& king);
    //LCP 50. 宝石补给
    int giveGem(vector<int>& gem, vector<vector<int>>& operations);
    //417. 太平洋大西洋水流问题
    int dir[4][2] = { -1, 0, 0, -1, 1, 0, 0, 1 };
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights);
    void pacificAtlantic_dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited, int x, int y);
    //LCP 06. 拿硬币
    int minCount(vector<int>& coins);
    //2603. 收集树中金币
    int collectTheCoins(vector<int>& coins, vector<vector<int>>& edges);
    //2591. 将钱分给最多的儿童
    int distMoney(int money, int children);
    //2546. 执行逐位运算使字符串相等
    bool makeStringsEqual(string s, string target);
    //2530. 执行K次操作后的最大分数
    long long maxKelements(vector<int>& nums, int k);
    //2597. 美丽子集的数目
    int beautifulSubsets(vector<int>& nums, int k);
    //260. 只出现一次的数字 III
    vector<int> singleNumberIII(vector<int>& nums);
    //2652. 倍数求和
    int sumOfMultiples(int n);
    //2643. 一最多的行
    vector<int> rowAndMaximumOnes(vector<vector<int>>& mat);
    //2611. 老鼠和奶酪
    int miceAndCheese(vector<int>& reward1, vector<int>& reward2, int k);
    //2544. 交替数字和
    int alternateDigitSum(int n);
    //2492. 两个城市间路径的最小分数
    int minScore(int n, vector<vector<int>>& roads);
    void minScore_dfs(unordered_set<int>& unionMap, vector<list<int>>& mapp, int n);
    //1726. 同积元组
    int tupleSameProduct(vector<int>& nums);
    void tupleSameProduct_Recursion(vector<int>& nums, unordered_map<int, int>& value_pair, vector<int>& curPair, int n);
    //1733. 需要教语言的最少人数
    int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships);
    //2525. 根据规则将箱子分类
    string categorizeBox(int length, int width, int height, int mass);
    //2596. 检查骑士巡视方案
    bool checkValidGrid(vector<vector<int>>& grid);
    //2316. 统计无向图中无法互相到达点对数
    long long countPairs(int n, vector<vector<int>>& edges);
    //2678. 老人的数目
    int countSeniors(vector<string>& details);
    //2740. 找出分区值
    int findValueOfPartition(vector<int>& nums);
    //1155. 掷骰子等于目标和的方法数
    int numRollsToTarget(int n, int k, int target);
    //1103. 分糖果II
    vector<int> distributeCandies(int candies, int num_people);
    //2698. 求一个整数的惩罚数
    int punishmentNumber(int n);
    bool punishmentNumber_Recursion(int num, string strNum, int index, int sum);
    //2749. 得到整数零需要执行的最少操作数
    int makeTheIntegerZero(int num1, int num2);
    //2520. 统计能整除数字的位数
    int countDigits(int num);
    //2581. 统计可能的树根数目
    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k);
    //1465. 切割后面积最大的蛋糕
    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts);
    //2558. 从数量最多的堆取走礼物
    long long pickGifts(vector<int>& gifts, int k);
    //2616. 最小化数对的最大差值
    int minimizeMax(vector<int>& nums, int p);
    //275. H指数II
    int hIndex(vector<int>& citations);
    //2003. 每棵子树内缺失的最小基因值
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums);
    //117. 填充每个节点的下一个右侧节点指针 II
    Node2* connect(Node2* root);
    void connect_Recursion(Node2* root, list<Node2*>& left, list<Node2*>& right);
    //217. 存在重复元素
    bool containsDuplicate(vector<int>& nums);
    //2609. 最长平衡子字符串
    int findTheLongestBalancedSubstring(string s);
    //2583. 二叉树中的第K大层和
    long long kthLargestLevelSum(TreeNode* root, int k);
    list<long long> kthLargestLevelSum_Recursion(TreeNode* root);
    //2258. 逃离火灾
    int maximumMinutes_ret;
    int maximumMinutes(vector<vector<int>>& grid);
    void maximumMinutes_Recursion(vector<vector<int>>& grid, vector<vector<bool>>& isLook, int x, int y, int curMin);
    //2300. 咒语和药水的成功对数
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success);
    //162. 寻找峰值
    int findPeakElement(vector<int>& nums);
    //2397. 被列覆盖的最多行数
    int maximumRows(vector<vector<int>>& matrix, int numSelect);
    //1944. 队列中可以看到的人数
    vector<int> canSeePersonsCount(vector<int>& heights);
    //2807. 在链表中插入最大公约数
    ListNode* insertGreatestCommonDivisors(ListNode* head);
    //447. 回旋镖的数量
    int numberOfBoomerangs(vector<vector<int>>& points);
    //2707. 字符串中的额外字符
    int minExtraChar(string s, vector<string>& dictionary);
    //2696. 删除子串后的字符串最小长度
    int minLength(string s);
    //2645. 构造有效字符串的最少插入数
    int addMinimum(string word);
    //2085. 统计出现过一次的公共字符串
    int countWords(vector<string>& words1, vector<string>& words2);
    //82. 删除排序链表中的重复元素II
    ListNode* deleteDuplicates(ListNode* head);
    //2719. 统计整数数目
    int count(string num1, string num2, int min_sum, int max_sum);
    //2376. 统计特殊整数--数位dp
    int countSpecialNumbers(int n);
    //233. 数字1的个数--数位dp
    int countDigitOne(int n);
    //902. 最大为N的数字组合--数位dp
    int atMostNGivenDigitSet(vector<string>& digits, int n);
    //600. 不含连续1的非负整数--数位dp
    int findIntegers(int n);
    //2788. 按分隔符拆分字符串
    vector<string> splitWordsBySeparator(vector<string>& words, char separator);
    //2865. 美丽塔I
    long long maximumSumOfHeights(vector<int>& maxHeights);
    //938. 二叉搜索树的范围和
    int rangeSumBST(TreeNode* root, int low, int high);
    //2867. 统计树中的合法路径数目
    const int countPaths_MAX = 1e5;
    bool* countPaths_isPrime;
    long long countPaths(int n, vector<vector<int>>& edges);
    //2673. 使二叉树所有路径值相等的最小代价
    int minIncrements(int n, vector<int>& cost);
    //2549. 统计桌面上的不同数字
    int distinctIntegers(int n);
    //834. 树中距离之和
    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges);
    //2369. 检查数组是否存在有效部分
    bool validPartition(vector<int>& nums);
    //2369. 受限条件下可到达节点的数目
    int reachableNodes(int n, vector<vector<int>>& edges, vector<int>& restricted);
    //2439. 最小化数组中的最大值--最大最小二分查找
    int minimizeArrayValue(vector<int>& nums);
    //2513. 最小化两个数组中的最大值--最大最小二分查找
    int minimizeSet(int divisor1, int divisor2, int uniqueCnt1, int uniqueCnt2);
    //1552. 两球之间的磁力--最大最小二分查找
    int maxDistance(vector<int>& position, int m);
    //2517. 礼盒的最大甜蜜度--最大最小二分查找
    int maximumTastiness(vector<int>& price, int k);
    //2528. 最大化城市的最小供电站数目--最大最小二分查找
    long long maxPower(vector<int>& stations, int r, int k);
    //1976. 到达目的地的方案数
    int countThePaths(int n, vector<vector<int>>& roads);
    //2917. 找出数组中的K-or值
    int findKOr(vector<int>& nums, int k);
    //2575. 找出字符串的可整除数组
    vector<int> divisibilityArray(string word, int m);
};
