#pragma once
#include "tool.h"

using namespace std;

class LClabuladong
{
public:
    LClabuladong() {}
    //208. ʵ�� Trie (ǰ׺��)
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
        //����ָ��word���һ���ַ�����ָ��
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
        //���ش�һ���ڵ㿪ʼ���ڵ����м�
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
        //����һ��ǰ׺�ļ���ֵsum
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


        //("themxyz")->them�����ǰ׺��them
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

        //("themxyz")->them���ǰ׺��them
        string longestPrefixOf(string query) {

        }

        //ǰ׺Ϊprefix�����м�
        vector<string> keysWithPrefix(string prefix) {
            vector<string> res;
            TrieNode* p = getTrieNode(this->root, prefix);
            if (p == NULL) {
                return res;
            }
            getAllKeysWithNode(p, res, prefix);
            return res;
        }

        //ǰ׺Ϊprefix�ļ��Ƿ����
        bool hasKeyWithPrefix(string prefix) {
            TrieNode* p = getTrieNode(this->root, prefix);
            return p != NULL;
        }

        //ͨ���'.'������ƥ������м�
        vector<string> keysWithPattern(string pattern) {

        }

        //ͨ���'.'���Ƿ����ƥ��ļ�
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
    //355. �������
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
    //46. ȫ����
    vector<vector<int>> res;
    void fullArrangement(list<int>& nums, vector<int>& path);
    vector<vector<int>> permute(vector<int>& nums);
    //322. ��Ǯ�һ�
    int* coinChange_memo;
    int coinChange_dp(vector<int>& coins, int amount);
    int coinChange(vector<int>& coins, int amount);
    //300. �����������
    int* lengthOfLIS_memo;
    int lengthOfLIS(vector<int>& nums);
    //354. ����˹�����ŷ�����
    static bool maxEnvelopes_cmp(const vector<int>& v1, const vector<int>& v2);
    int maxEnvelopes(vector<vector<int>>& envelopes);
    //931. �½�·����С��
    int minFallingPathSum(vector<vector<int>>& matrix);
    //72. �༭����
    vector<vector<int>> minDistance_memo;
    int minDistance_min(int a, int b, int c);
    int minDistance_dp(string word1, int i, string word2, int j);
    int minDistance(string word1, string word2);
    int minDistance2(string word1, string word2);
    //53. ����������
    int maxSubArray_dp(vector<int>& nums);
    int maxSubArray_SlideWindow(vector<int>& nums);
    int maxSubArray_PrefixSum(vector<int>& nums);
    //1143. �����������
    int longestCommonSubsequence(string text1, string text2);
    //583. �����ַ�����ɾ������
    int minDeleteDistance(string word1, string word2);
    //712. �����ַ�������СASCIIɾ����
    int minimumDeleteSum(string s1, string s2);
    //1312. ���ַ�����Ϊ���Ĵ������ٲ������
    int minInsertions(string s);
    //516. �����������
    int longestPalindromeSubseq(string s);
    //416. �ָ�Ⱥ��Ӽ�
    bool canPartition(vector<int>& nums);
    //12. ����ת��������
    void addRomanNum(int num, char arr_roman, string& res);
    string intToRoman(int num);
    //49. ��ĸ��λ�ʷ���
    vector<vector<string>> groupAnagrams(vector<string>& strs);
    //62. ��ͬ·��
    int uniquePaths(int m, int n);
    //131. �ָ���Ĵ�
    string s_val;
    vector<vector<string>> partitionPalindromic_res;
    void startPartition(vector<vector<bool>>& memo, int begin, vector<string>& strs);
    vector<vector<string>> partitionPalindromic(string s);
    //494. Ŀ���
    int findTargetSumWays(vector<int>& nums, int target);
    //279. ��ȫƽ����
    int numSquares(int n);
    //377. ����ܺ� IV
    int combinationSum4(vector<int>& nums, int target);
    //518. ��Ǯ�һ� II
    int coinChangeII(int amount, vector<int>& coins);
    //64. ��С·����
    int minPathSum(vector<vector<int>>& grid);
    //174. ���³���Ϸ
    int calculateMinimumHP(vector<vector<int>>& dungeon);
    //514. ����֮·
    unordered_map<char, list<int>> findRotateSteps_char_to_index;
    vector<vector<int>> findRotateSteps_memo;
    int findRotateSteps(string ring, string key);
    int findRotateSteps_dp(string ring, int i, string key, int j);
    //17. �绰�������ĸ���
    vector<string> letterCombinations_ret;
    unordered_map<int, string> letterCombinations_digit_to_letter;
    vector<string> letterCombinations(string digits);
    void digits2letters(string digits, int begin, string res);
    //24. �������������еĽڵ�
    ListNode* swapPairs(ListNode* head);
    //99. �ָ�����������
    vector<int> recoverTree_res;
    void recoverTree(TreeNode* root);
    void recoverTree_dfs(TreeNode* root, vector<TreeNode*>& order);
    //51. N �ʺ�
    vector<vector<string>> solveNQueens_res;
    vector<vector<string>> solveNQueens(int n);
    void solveNQueens_recursion(vector<string>& chessboard, int queen_order);
    bool solveNQueens_isAvailable(vector<string>& chessboard, int row, int col);
    //698. ����Ϊk����ȵ��Ӽ�
    bool canPartitionKSubsets(vector<int>& nums, int k);
    bool canPartitionKSubsets_loadBuckets(vector<int>& buckets, int bucket_index, vector<int>& nums, int nums_index);
    //200. ��������
    int numIslands(vector<vector<char>>& grid);
    void numIslands_connect(vector<vector<char>>& grid, int i, int j);
    //1254. ͳ�Ʒ�յ������Ŀ
    int closedIsland(vector<vector<int>>& grid);
    void closedIsland_isClosedIsland(vector<vector<int>>& grid, int i, int j);
    void closedIsland_reverse(vector<vector<int>>& grid, int i, int j);
    //1020. �ɵص�����
    int numEnclaves(vector<vector<int>>& grid);
    void numEnclaves_isClosed(vector<vector<int>>& grid, int i, int j, int& ret);
    void numEnclaves_reverse(vector<vector<int>>& grid, int i, int j);
    //695. �����������
    int maxAreaOfIsland(vector<vector<int>>& grid);
    int maxAreaOfIsland_countArea(vector<vector<int>>& grid, int i, int j);
    //1905. ͳ���ӵ���
    int countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2);
    void countSubIslands_islandReverse(vector<vector<int>>& grid, int i, int j, int pave, int val);
    //111. ����������С���
    int minDepth(TreeNode* root);
    //752. ��ת����
    int openLock(vector<string>& deadends, string target);
    //773. ��������
    int slidingPuzzle(vector<vector<int>>& board);
    bool slidingPuzzle_isOK(vector<vector<int>>& board);
    void slidingPuzzle_swapTwoSquare(vector<vector<int>>& board, int i1, int j1, int i2, int j2);
    pair<int, int> slidingPuzzle_findZero(vector<vector<int>>& board);
    bool slidingPuzzle_isVisited(vector<vector<int>>& board, bool visited[6][6][6][6][6][6]);
    void slidingPuzzle_visit(vector<vector<int>>& board, bool visited[6][6][6][6][6][6]);
    //787. K վ��ת������˵ĺ���
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k);
    vector<vector<int>> findCheapestPrice_creatPriceTable(vector<vector<int>>& flights, int n);
    //486. Ԥ��Ӯ��
    bool PredictTheWinner(vector<int>& nums);
    //299. ��������Ϸ
    string getHint(string secret, string guess);
    //238. ��������������ĳ˻�
    vector<int> productExceptSelf(vector<int>& nums);
    //1922. ͳ�ƺ����ֵ���Ŀ
    int countGoodNumbers(long long n);
    //139. ���ʲ��
    bool wordBreak(string s, vector<string>& wordDict);
    //43. �ַ������
    string multiply(string num1, string num2);
    vector<int> multiply_add(vector<int> num1, vector<int> num2);
    //400. �� N λ����
    int findNthDigit(int n);
    //390. ������Ϸ
    int lastRemaining(int n);
    //789. �����谭��
    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target);
    int escapeGhosts_distance(vector<int>& source, vector<int>& target);
    //873. ���쳲����������еĳ���
    int lenLongestFibSubseq(vector<int>& arr);
    //1262. �ɱ�������������
    int maxSumDivThree(vector<int>& nums);
    //688. ��ʿ�������ϵĸ���
    double knightProbability(int n, int k, int row, int column);
    double knightProbability_getProbability(int n, int x, int y, vector<vector<double>>& dp);
    //967. ��������ͬ������
    vector<int> numsSameConsecDiff(int n, int k);
    void numsSameConsecDiff_recursion(int n, int k, int p, vector<int>& cur);
    vector<int> numsSameConsecDiff_ret;
    //10. ������ʽƥ��
    bool isMatch(string s, string p);
    //887. ��������
    unordered_map<int, int> superEggDrop_memo;
    int superEggDrop(int k, int n);
    int superEggDrop_dp(int k, int n);
    //877. ʯ����Ϸ
    bool stoneGame(vector<int>& piles);
    //198. ��ҽ���
    int rob(vector<int>& nums);
    //��������
    int MachineHandling(int n, vector<int> d1, vector<int> d2);
    void MachineHandling_judgement(int t1, int t2, int& temp1, int& temp2);
    //871. ��ͼ��ʹ���
    int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations);
    //213. ��ҽ��� II
    int robII(vector<int>& nums);
    //337. ��ҽ��� III
    int robIII(TreeNode* root);
    int robIII_cursion(TreeNode* root);
    //2560. ��ҽ��� IV
    int minCapability(vector<int>& nums, int k);
    bool minCapability_check(vector<int>& nums, int k, int m);
    //121. ������Ʊ�����ʱ��
    int maxProfit(vector<int>& prices);
    //122. ������Ʊ�����ʱ�� II
    int maxProfitII(vector<int>& prices);
    //123. ������Ʊ�����ʱ�� III
    int maxProfitIII(vector<int>& prices);
    //66. ��һ
    vector<int> plusOne(vector<int>& digits);
    //188. ������Ʊ�����ʱ�� IV
    int maxProfitIV(int k, vector<int>& prices);
    //309. ���������Ʊʱ�����䶳��
    int maxProfitFreezing(vector<int>& prices);
    //714. ������Ʊ�����ʱ����������
    int maxProfitV(vector<int>& prices, int fee);
    //8. �ҳ��ַ����е�һ��ƥ������±�
    int strStr(string haystack, string needle);
    int strStr_getIndex(string& needle, vector<int>& kmp_arr, int cur);
    //435. ���ص�����
    int eraseOverlapIntervals(vector<vector<int>>& intervals);
    //1024. ��Ƶƴ��
    int videoStitching(vector<vector<int>>& clips, int time);
    //45. ��Ծ��Ϸ II
    int jump(vector<int>& nums);
    //55. ��Ծ��Ϸ
    bool canJump(vector<int>& nums);
    //52. N �ʺ� II
    int totalNQueens(int n);
    void totalNQueens_recursion(int row);
    bool totalNQueens_mapIsOk(int row, int col);
    int totalNQueens_ret;
    vector<vector<char>> totalNQueens_map;
    //216. ����ܺ� III
    vector<vector<int>> combinationSum3(int k, int n);
    void combinationSum3_recursion(int k, int n, vector<vector<int>>& buckets, int& v_sum, vector<int>& cur, int x);
    //40. ����ܺ� II
    vector<vector<int>> combinationSum2_ret;
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
    void combinationSum2_backtrace(vector<int>& candidates, int target, vector<int>& cur, int sum, int x);
    //47. ȫ���� II
    vector<vector<int>> permuteUnique_ret;
    vector<vector<int>> permuteUnique(vector<int>& nums);
    void permuteUnique_backtrace(vector<int>& nums, vector<int>& cur, vector<bool>& nums_flags, int x);
    //90. �Ӽ� II
    vector<vector<int>> subsetsWithDup_ret;
    vector<vector<int>> subsetsWithDup(vector<int>& nums);
    void subsetsWithDup_backtrace(vector<int>& nums, vector<int>& cur, int x);
    //37. ������
    vector<vector<char>> solveSudoku_ret;
    bool solveSudoku_IsOk;
    void solveSudoku(vector<vector<char>>& board);
    void solveSuduku_InitGrid(vector<vector<char>>& board, vector<vector<bool>>& grid_flag);
    void solveSuduku_BackTrace(vector<vector<char>>& board, vector<vector<bool>>& grid_flag, int grid_num, int number);
    bool solveSuduku_RemedyIsOk(vector<vector<char>>& board, int x, int y, int value);
    //22. ��������
    vector<string> generateParenthesis_ret;
    vector<string> generateParenthesis(int n);
    void generateParenthesis_BackTrace(int n, int index, string& cur, int left_num, int right_num);
    //382. ��������ڵ�
    ListNode* getRandom_head;
    LClabuladong(ListNode* head);
    int getRandom();
    //384. ��������
    vector<int> shuffle_arr;
    LClabuladong(vector<int>& nums);
    vector<int> reset();
    vector<int> shuffle();
    //398. ���������
    int RandomPick(int target);
    //136. ֻ����һ�ε�����
    int singleNumber(vector<int>& nums);
    //191. λ1�ĸ���
    int hammingWeight(uint32_t n);
    //231. 2 ����
    bool isPowerOfTwo(int n);
    //268. ��ʧ������
    int missingNumber(vector<int>& nums);
    //172. �׳˺����
    long long trailingZeroes(long long n);
    //793. �׳˺����� K ����
    int preimageSizeFZF(int k);
    long long preimageSizeFZF_LeftBound(int k);
    long long preimageSizeFZF_rightBound(int k);
    //204. ��������
    int countPrimes(int n);
    //372. �����η�
    int superPow_mod = 1337;
    int superPow(int a, vector<int>& b);
    int pow1(int x, int n);
    //645. ����ļ���
    vector<int> findErrorNums(vector<int>& nums);
    //292. Nim ��Ϸ
    bool canWinNim(int n);
    //319. ���ݿ���
    int bulbSwitch(int n);
    //241. Ϊ������ʽ������ȼ�
    vector<int> diffWaysToCompute_ret;
    vector<int> diffWaysToCompute(string expression);
    vector<int> diffWaysToCompute_GetComupute(vector<string> expression, int begin, int end);
    vector<string> diffWaysToCompute_GetOpVector(string expression);
    //1288. ɾ������������
    int removeCoveredIntervals(vector<vector<int>>& intervals);
    //56. �ϲ�����
    vector<vector<int>> merge(vector<vector<int>>& intervals);
    //986. �����б�Ľ���
    vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList);
    //659. �ָ�����Ϊ����������
    bool isPossible(vector<int>& nums);
    //969. �������
    vector<int> pancakeSort(vector<int>& arr);
    //224. ����������
    int calculate(string s);
    //42. ����ˮ
    int trap(vector<int>& height);
    int trap_recalculation(vector<int> height, int startIndex, int endIndex);
    //391. ��������
    bool isRectangleCover(vector<vector<int>>& rectangles);
    //855. ��������
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
    //392. �ж�������
    bool isSubsequence(string s, string t);
    //792. ƥ�������еĵ�����
    int numMatchingSubseq(string s, vector<string>& words);
    int numMatchingSubseq_FindNext(vector<int>& charRecord, int count);
    //977. ���������ƽ��
    vector<int> sortedSquares(vector<int>& nums);
    //367. ��Ч����ȫƽ����
    bool isPerfectSquare(int num);
    //844. �ȽϺ��˸���ַ���
    bool backspaceCompare(string s, string t);
    //904. ˮ������
    int totalFruit(vector<int>& fruits);
    //59. �������� II
    vector<vector<int>> generateMatrix(int n);
    void generateMatrix_Padding(vector<vector<int>>& matrix, int xStart, int yStart, int value, int n);
    //54. ��������
    vector<int> spiralOrder(vector<vector<int>>& matrix);
    //203. �Ƴ�����Ԫ��
    ListNode* removeElements(ListNode* head, int val);
    //707. �������
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
    //242. ��Ч����ĸ��λ��
    bool isAnagram(string s, string t);
    //383. �����
    bool canConstruct(string ransomNote, string magazine);
    //349. ��������Ľ���
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2);
    //350. ��������Ľ��� II
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2);
    //202. ������
    bool isHappy(int n);
    //1. ����֮��
    vector<int> twoSum(vector<int>& nums, int target);
    //454. ������� II
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4);
    //15. ����֮��
    vector<vector<int>> threeSum(vector<int>& nums);
    //18. ����֮��
    vector<vector<int>> fourSum(vector<int>& nums, int target);
    //344. ��ת�ַ���
    void reverseString(vector<char>& s);
    //541. ��ת�ַ��� II
    string reverseStr(string s, int k);
    //151. ��ת�ַ����еĵ���
    string reverseWords(string s);
    vector<string> reverseWords_GetWords(string s);
    //459. �ظ������ַ���
    bool repeatedSubstringPattern(string s);
    //1047. ɾ���ַ����е����������ظ���
    string removeDuplicates(string s);
    //150. �沨�����ʽ��ֵ
    int evalRPN(vector<string>& tokens);
    int evalRPN_Compute(int a, int b, char op);
    //347. ǰ K ����ƵԪ��
    vector<int> topKFrequent(vector<int>& nums, int k);
    //145. �������ĺ������
    vector<int> postorderTraversal_ret;
    vector<int> postorderTraversal(TreeNode* root);
    void postorderTraversal_recursion(TreeNode* root);
    //102. �������Ĳ������
    vector<vector<int>> levelOrder(TreeNode* root);
    //589. N ������ǰ�����
    vector<int> preorder(Node* root);
    //590. N �����ĺ������
    vector<int> postorder(Node* root);
    //101. �Գƶ�����
    bool isSymmetric(TreeNode* root);
    bool isSymmetric_Compare(TreeNode* left, TreeNode* right);
    //100. ��ͬ����
    bool isSameTree(TreeNode* p, TreeNode* q);
    //572. ��һ����������
    bool isSubtree(TreeNode* root, TreeNode* subRoot);
    bool isSubtree_Contain(TreeNode* root, TreeNode* subRoot);
    //110. ƽ�������
    bool isBalanced(TreeNode* root);
    int isBalanced_GetDiff(TreeNode* root);
    //257. ������������·��
    vector<string> binaryTreePaths(TreeNode* root);
    vector<string> binaryTreePaths_GetPath(TreeNode* root);
    //404. ��Ҷ��֮��
    int sumOfLeftLeaves_ret;
    int sumOfLeftLeaves(TreeNode* root);
    void sumOfLeftLeaves_Recursion(TreeNode* root);
    //513. �������½ǵ�ֵ
    int findBottomLeftValue_h;
    int findBottomLeftValue_ret;
    int findBottomLeftValue(TreeNode* root);
    void findBottomLeftValue_Recursion(TreeNode* root, int h);
    //112. ·���ܺ�
    bool hasPathSum(TreeNode* root, int targetSum);
    bool hasPathSum_Recursion(TreeNode* root, int targetSum);
    //530. ��������������С���Բ�
    int getMinimumDifference(TreeNode* root);
    int getMinimumDifference_Recursion(TreeNode* root);
    int getMinimumDifference_GetMax(TreeNode* root);
    int getMinimumDifference_GetMin(TreeNode* root);
    //501. �����������е�����
    vector<int> findMode(TreeNode* root);
    void findMode_TreeToVector(TreeNode* root, vector<int>& nodeList);
    //108. ����������ת��Ϊ����������
    TreeNode* sortedArrayToBST(vector<int>& nums);
    TreeNode* sortedArrayToBST_Traverse(vector<int>& nums);
    //93. ��ԭ IP ��ַ
    vector<string> restoreIpAddresses_ret;
    vector<string> restoreIpAddresses(string s);
    void restoreIpAddresses_Traverse(string s, string cur, int n);
    //491. ����������
    vector<vector<int>> findSubsequences_ret;
    vector<vector<int>> findSubsequences(vector<int>& nums);
    void findSubsequences_Traverse(vector<int>& nums, int begin, vector<int> cur);
    //332. ���°����г�
    unordered_map<string, map<string, int>> findItinerary_targets;
    vector<string> findItinerary(vector<vector<string>>& tickets);
    bool findItinerary_Traverse(int ticketNum, vector<string>& result);
    //455. �ַ�����
    int findContentChildren(vector<int>& g, vector<int>& s);
    //376. �ڶ�����
    int wiggleMaxLength(vector<int>& nums);
    //1005. K ��ȡ������󻯵������
    int largestSumAfterKNegations(vector<int>& nums, int k);
    //134. ����վ
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost);
    //135. �ַ��ǹ�
    int candy(vector<int>& ratings);
    //860. ����ˮ����
    bool lemonadeChange(vector<int>& bills);
    //406. ��������ؽ�����
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people);
    //452. �����������ļ���������
    int findMinArrowShots(vector<vector<int>>& points);
    //763. ������ĸ����
    vector<int> partitionLabels(string s);
    //738. ��������������
    int monotoneIncreasingDigits(int n);
    //968. ��ض�����
    int minCameraCover_ret;
    int minCameraCover(TreeNode* root);
    int minCameraCover_recursion(TreeNode* cur);
    //746. ʹ����С������¥��
    int minCostClimbingStairs(vector<int>& cost);
    //63. ��ͬ·�� II
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
    //343. �������
    int integerBreak(int n);
    //1049. ���һ��ʯͷ������ II
    int lastStoneWeightII(vector<int>& stones);
    //474. һ����
    int findMaxForm(vector<string>& strs, int m, int n);
    pair<int, int> findMaxForm_CountZeroOne(string s);
    //674. �������������
    int findLengthOfLCIS(vector<int>& nums);
    //718. ��ظ�������
    int findLength(vector<int>& nums1, vector<int>& nums2);
    //1035. ���ཻ����
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2);
    //115. ��ͬ��������
    int numDistinct(string s, string t);
    //647. �����Ӵ�
    int countSubstrings(string s);
    //84. ��״ͼ�����ľ���
    int largestRectangleArea(vector<int>& heights);
    //2605. ����������������������С����
    int minNumber(vector<int>& nums1, vector<int>& nums2);
    //1123. ����Ҷ�ڵ�������������
    TreeNode* lcaDeepestLeaves(TreeNode* root);
    void lcaDeepestLeaves_CountDeep(TreeNode* root, int d, map<int, vector<TreeNode*>>& deepNodes);
    int lcaDeepestLeaves_FindParents(TreeNode* root, int n, TreeNode*& res);
    //2594. �޳�������ʱ��
    long long repairCars(vector<int>& ranks, int cars);
    bool repairCars_CanRepair(vector<int>& ranks, int cars, long long time);
    //2651. �����г���վʱ��
    int findDelayedArrivalTime(int arrivalTime, int delayedTime);
    //1222. ���Թ��������Ļʺ�
    vector<vector<int>> queensAttacktheKing(vector<vector<int>>& queens, vector<int>& king);
    //LCP 50. ��ʯ����
    int giveGem(vector<int>& gem, vector<vector<int>>& operations);
    //417. ̫ƽ�������ˮ������
    int dir[4][2] = { -1, 0, 0, -1, 1, 0, 0, 1 };
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights);
    void pacificAtlantic_dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited, int x, int y);
    //LCP 06. ��Ӳ��
    int minCount(vector<int>& coins);
    //2603. �ռ����н��
    int collectTheCoins(vector<int>& coins, vector<vector<int>>& edges);
    //2591. ��Ǯ�ָ����Ķ�ͯ
    int distMoney(int money, int children);
    //2546. ִ����λ����ʹ�ַ������
    bool makeStringsEqual(string s, string target);
    //2530. ִ��K�β������������
    long long maxKelements(vector<int>& nums, int k);
    //2597. �����Ӽ�����Ŀ
    int beautifulSubsets(vector<int>& nums, int k);
    //260. ֻ����һ�ε����� III
    vector<int> singleNumberIII(vector<int>& nums);
    //2652. �������
    int sumOfMultiples(int n);
    //2643. һ������
    vector<int> rowAndMaximumOnes(vector<vector<int>>& mat);
    //2611. ���������
    int miceAndCheese(vector<int>& reward1, vector<int>& reward2, int k);
    //2544. �������ֺ�
    int alternateDigitSum(int n);
    //2492. �������м�·������С����
    int minScore(int n, vector<vector<int>>& roads);
    void minScore_dfs(unordered_set<int>& unionMap, vector<list<int>>& mapp, int n);
    //1726. ͬ��Ԫ��
    int tupleSameProduct(vector<int>& nums);
    void tupleSameProduct_Recursion(vector<int>& nums, unordered_map<int, int>& value_pair, vector<int>& curPair, int n);
    //1733. ��Ҫ�����Ե���������
    int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships);
    //2525. ���ݹ������ӷ���
    string categorizeBox(int length, int width, int height, int mass);
    //2596. �����ʿѲ�ӷ���
    bool checkValidGrid(vector<vector<int>>& grid);
    //2316. ͳ������ͼ���޷����ൽ������
    long long countPairs(int n, vector<vector<int>>& edges);
    //2678. ���˵���Ŀ
    int countSeniors(vector<string>& details);
    //2740. �ҳ�����ֵ
    int findValueOfPartition(vector<int>& nums);
    //1155. �����ӵ���Ŀ��͵ķ�����
    int numRollsToTarget(int n, int k, int target);
    //1103. ���ǹ�II
    vector<int> distributeCandies(int candies, int num_people);
    //2698. ��һ�������ĳͷ���
    int punishmentNumber(int n);
    bool punishmentNumber_Recursion(int num, string strNum, int index, int sum);
    //2749. �õ���������Ҫִ�е����ٲ�����
    int makeTheIntegerZero(int num1, int num2);
    //2520. ͳ�����������ֵ�λ��
    int countDigits(int num);
    //2581. ͳ�ƿ��ܵ�������Ŀ
    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k);
    //1465. �и��������ĵ���
    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts);
    //2558. ���������Ķ�ȡ������
    long long pickGifts(vector<int>& gifts, int k);
    //2616. ��С�����Ե�����ֵ
    int minimizeMax(vector<int>& nums, int p);
    //275. Hָ��II
    int hIndex(vector<int>& citations);
    //2003. ÿ��������ȱʧ����С����ֵ
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums);
    //117. ���ÿ���ڵ����һ���Ҳ�ڵ�ָ�� II
    Node2* connect(Node2* root);
    void connect_Recursion(Node2* root, list<Node2*>& left, list<Node2*>& right);
    //217. �����ظ�Ԫ��
    bool containsDuplicate(vector<int>& nums);
    //2609. �ƽ�����ַ���
    int findTheLongestBalancedSubstring(string s);
    //2583. �������еĵ�K����
    long long kthLargestLevelSum(TreeNode* root, int k);
    list<long long> kthLargestLevelSum_Recursion(TreeNode* root);
    //2258. �������
    int maximumMinutes_ret;
    int maximumMinutes(vector<vector<int>>& grid);
    void maximumMinutes_Recursion(vector<vector<int>>& grid, vector<vector<bool>>& isLook, int x, int y, int curMin);
    //2300. �����ҩˮ�ĳɹ�����
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success);
    //162. Ѱ�ҷ�ֵ
    int findPeakElement(vector<int>& nums);
    //2397. ���и��ǵ��������
    int maximumRows(vector<vector<int>>& matrix, int numSelect);
    //1944. �����п��Կ���������
    vector<int> canSeePersonsCount(vector<int>& heights);
    //2807. �������в������Լ��
    ListNode* insertGreatestCommonDivisors(ListNode* head);
    //447. �����ڵ�����
    int numberOfBoomerangs(vector<vector<int>>& points);
    //2707. �ַ����еĶ����ַ�
    int minExtraChar(string s, vector<string>& dictionary);
    //2696. ɾ���Ӵ�����ַ�����С����
    int minLength(string s);
    //2645. ������Ч�ַ��������ٲ�����
    int addMinimum(string word);
    //2085. ͳ�Ƴ��ֹ�һ�εĹ����ַ���
    int countWords(vector<string>& words1, vector<string>& words2);
    //82. ɾ�����������е��ظ�Ԫ��II
    ListNode* deleteDuplicates(ListNode* head);
    //2719. ͳ��������Ŀ
    int count(string num1, string num2, int min_sum, int max_sum);
    //2376. ͳ����������--��λdp
    int countSpecialNumbers(int n);
    //233. ����1�ĸ���--��λdp
    int countDigitOne(int n);
    //902. ���ΪN���������--��λdp
    int atMostNGivenDigitSet(vector<string>& digits, int n);
    //600. ��������1�ķǸ�����--��λdp
    int findIntegers(int n);
    //2788. ���ָ�������ַ���
    vector<string> splitWordsBySeparator(vector<string>& words, char separator);
    //2865. ������I
    long long maximumSumOfHeights(vector<int>& maxHeights);
    //938. �����������ķ�Χ��
    int rangeSumBST(TreeNode* root, int low, int high);
    //2867. ͳ�����еĺϷ�·����Ŀ
    const int countPaths_MAX = 1e5;
    bool* countPaths_isPrime;
    long long countPaths(int n, vector<vector<int>>& edges);
    //2673. ʹ����������·��ֵ��ȵ���С����
    int minIncrements(int n, vector<int>& cost);
    //2549. ͳ�������ϵĲ�ͬ����
    int distinctIntegers(int n);
    //834. ���о���֮��
    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges);
    //2369. ��������Ƿ������Ч����
    bool validPartition(vector<int>& nums);
    //2369. ���������¿ɵ���ڵ����Ŀ
    int reachableNodes(int n, vector<vector<int>>& edges, vector<int>& restricted);
    //2439. ��С�������е����ֵ--�����С���ֲ���
    int minimizeArrayValue(vector<int>& nums);
    //2513. ��С�����������е����ֵ--�����С���ֲ���
    int minimizeSet(int divisor1, int divisor2, int uniqueCnt1, int uniqueCnt2);
    //1552. ����֮��Ĵ���--�����С���ֲ���
    int maxDistance(vector<int>& position, int m);
    //2517. ��е�������۶�--�����С���ֲ���
    int maximumTastiness(vector<int>& price, int k);
    //2528. ��󻯳��е���С����վ��Ŀ--�����С���ֲ���
    long long maxPower(vector<int>& stations, int r, int k);
    //1976. ����Ŀ�ĵصķ�����
    int countThePaths(int n, vector<vector<int>>& roads);
    //2917. �ҳ������е�K-orֵ
    int findKOr(vector<int>& nums, int k);
    //2575. �ҳ��ַ����Ŀ���������
    vector<int> divisibilityArray(string word, int m);
};
