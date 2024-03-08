#include "LClabuladong.h"

LClabuladong::Twitter::User::User(int id)
{
	user_id = id;
	head = NULL;
	followed.push_back(this);
}

void LClabuladong::Twitter::User::post(PersonTwitter* twee)
{
	twee->next = head;
	head = twee;
}

void LClabuladong::Twitter::User::follow(User* followee)
{
	for (User* user : followed) {
		if (followee->user_id == user->user_id) {
			return;
		}
	}
	followed.push_back(followee);
}

void LClabuladong::Twitter::User::unfollow(User* unfollowee)
{
	followed.remove(unfollowee);
}

vector<int> LClabuladong::Twitter::User::getNewTwitter()
{
	vector<int> res;
	list<PersonTwitter*> ptr_twee;
	for (list<User*>::iterator i = followed.begin(); i != followed.end(); i++) {
		if ((*i)->head) {
			ptr_twee.push_back((*i)->head);
		}
	}
	int max_time;
	int max_id;
	PersonTwitter** ptr_p = NULL;
	while (!ptr_twee.empty()) {
		max_time = -1;
		max_id = -1;
		for (PersonTwitter*& twee : ptr_twee) {
			if (twee->time > max_time) {
				max_time = twee->time;
				max_id = twee->id;
				ptr_p = &twee;
			}
		}
		res.push_back(max_id);
		(*ptr_p) = (*ptr_p)->next;
		if ((*ptr_p) == NULL) {
			ptr_twee.remove(*ptr_p);
		}
	}
	if (res.size() > 10) {
		res.resize(10);
	}

	return res;
}

LClabuladong::Twitter::PersonTwitter::PersonTwitter(int i, int timestamp)
{
	id = i;
	time = timestamp;
	next = NULL;
}

LClabuladong::Twitter::Twitter()
{
	timestamp = 0;
}

void LClabuladong::Twitter::postTweet(int userId, int tweetId)
{
	if (!users.count(userId)) {
		User* temp_user = new User(userId);
		users.insert(pair<int, User*>(userId, temp_user));
	}
	PersonTwitter* temp_twitter = new PersonTwitter(tweetId, timestamp++);
	users[userId]->post(temp_twitter);
}

vector<int> LClabuladong::Twitter::getNewsFeed(int userId)
{
	if (!users.count(userId)) {
		User* temp_user = new User(userId);
		users.insert(pair<int, User*>(userId, temp_user));
	}
	return users[userId]->getNewTwitter();
}

void LClabuladong::Twitter::follow(int followerId, int followeeId)
{
	if (!users.count(followerId)) {
		User* temp_user = new User(followerId);
		users.insert(pair<int, User*>(followerId, temp_user));
	}
	if (!users.count(followeeId)) {
		User* temp_user = new User(followeeId);
		users.insert(pair<int, User*>(followeeId, temp_user));
	}
	users[followerId]->follow(users[followeeId]);
}

void LClabuladong::Twitter::unfollow(int followerId, int followeeId)
{
	users[followerId]->unfollow(users[followeeId]);
}

void LClabuladong::fullArrangement(list<int>& nums, vector<int>& path)
{
	if (nums.empty()) {
		res.push_back(path);
		return;
	}
	int flag = nums.size();
	while (flag != 0) {
		int i = nums.front();
		path.push_back(i);
		nums.pop_front();
		fullArrangement(nums, path);
		path.pop_back();
		nums.push_back(i);
		--flag;
	}
}

vector<vector<int>> LClabuladong::permute(vector<int>& nums)
{
	list<int> myl;
	for (int i = 0; i < nums.size(); i++) {
		myl.push_back(nums[i]);
	}
	vector<int> path;
	fullArrangement(myl, path);
	return res;
}

int LClabuladong::coinChange_dp(vector<int>& coins, int amount)
{
	if (amount == 0) {
		return 0;
	}
	if (amount < 0) {
		return -1;
	}
	if (coinChange_memo[amount] != -2) {
		return coinChange_memo[amount];
	}
	int res = INT_MAX;
	for (int coin : coins) {
		int sub_pro = amount - coin;
		int sub_least = coinChange_dp(coins, sub_pro);

		if (sub_least == -1) {
			continue;
		}
		res = min(res, sub_least + 1);
	}
	coinChange_memo[amount] = (res == INT_MAX ? -1 : res);
	return coinChange_memo[amount];
}

int LClabuladong::coinChange(vector<int>& coins, int amount)
{
	coinChange_memo = new int[amount + 1]();
	for (int i = 0; i < amount + 1; i++) {
		coinChange_memo[i] = -2;
	}
	return coinChange_dp(coins, amount);
}

int LClabuladong::lengthOfLIS(vector<int>& nums)
{
	lengthOfLIS_memo = new int[nums.size() + 1]();
	for (int i = 0; i < nums.size() + 1; i++) {
		lengthOfLIS_memo[i] = -1;
	}
	lengthOfLIS_memo[1] = 1;
	for (int i = 2; i <= nums.size(); i++) {
		int sub_longest = 1;
		for (int j = 1; j < i; j++) {
			if (nums[j - 1] < nums[i - 1]) {
				sub_longest = max(sub_longest, 1 + lengthOfLIS_memo[j]);
			}
		}
		lengthOfLIS_memo[i] = sub_longest;
	}
	int res = INT_MIN;
	for (int i = 1; i <= nums.size(); i++) {
		res = max(res, lengthOfLIS_memo[i]);
	}

	return res;
}

int LClabuladong::maxEnvelopes(vector<vector<int>>& envelopes)
{
	sort(envelopes.begin(), envelopes.end(), maxEnvelopes_cmp);

	vector<int> top(envelopes.size(), -1);
	int pile = 0;
	for (int i = 0; i < envelopes.size(); i++) {
		int poker = envelopes[i][1];
		int left = 0, right = pile;//[0, pile)
		while (left < right) {
			int mid = (left + right) / 2;
			if (top[mid] > poker) {
				right = mid;
			}
			else if (top[mid] < poker) {
				left = mid + 1;
			}
			else if (top[mid] == poker) {
				right = mid;
			}
		}
		if (left == pile) ++pile;
		top[left] = poker;
	}

	return pile;
}

bool LClabuladong::maxEnvelopes_cmp(const vector<int>& v1, const vector<int>& v2)
{
	if (v1[0] == v2[0]) return v1[1] > v2[1];
	return v1[0] < v2[0];
}

int LClabuladong::minFallingPathSum(vector<vector<int>>& matrix)
{
	vector<vector<int>> memo(2, vector<int>(matrix[0].size(), INT_MAX));
	int height = matrix.size(), weight = matrix[0].size();
	for (int i = 0; i < weight; i++) {
		memo[1][i] = matrix[height - 1][i];
	}
	for (int i = height - 2; i >= 0; i--) {
		for (int j = 0; j < weight; j++) {
			if (j - 1 >= 0) {
				memo[0][j] = min(memo[0][j], matrix[i][j] + memo[1][j - 1]);
			}
			if (j + 1 < weight) {
				memo[0][j] = min(memo[0][j], matrix[i][j] + memo[1][j + 1]);
			}
			memo[0][j] = min(memo[0][j], matrix[i][j] + memo[1][j]);
		}
		for (int k = 0; k < weight; k++) {
			memo[1][k] = memo[0][k];
			memo[0][k] = INT_MAX;
		}
	}
	int res = INT_MAX;
	for (int i = 0; i < weight; i++) {
		res = min(memo[1][i], res);
	}
	return res;
}

int LClabuladong::minDistance(string word1, string word2)
{
	int len1 = word1.size(), len2 = word2.size();
	minDistance_memo.resize(len1, vector<int>(len2, -1));

	return minDistance_dp(word1, len1 - 1, word2, len2 - 1);
}

int LClabuladong::minDistance_dp(string word1, int i, string word2, int j)
{
	if (i == -1 && j == -1) {
		return 0;
	}
	if (i == -1) {
		return minDistance_dp(word1, i, word2, j - 1) + 1;
	}
	if (j == -1) {
		return minDistance_dp(word1, i - 1, word2, j) + 1;
	}

	if (minDistance_memo[i][j] != -1) {
		return minDistance_memo[i][j];
	}

	if (word1[i] == word2[j]) {
		minDistance_memo[i][j] = minDistance_dp(word1, i - 1, word2, j - 1);
	}
	else {
		minDistance_memo[i][j] = minDistance_min(
			minDistance_dp(word1, i, word2, j - 1) + 1,//插入
			minDistance_dp(word1, i - 1, word2, j) + 1,//删除
			minDistance_dp(word1, i - 1, word2, j - 1) + 1//替换
		);
	}

	return minDistance_memo[i][j];
}

int LClabuladong::minDistance_min(int a, int b, int c)
{
	return min(a, min(b, c));
}

int LClabuladong::minDistance2(string word1, string word2)
{
	int len2 = word1.size(), len1 = word2.size();
	minDistance_memo.resize(len1 + 1, vector<int>(len2 + 1, INT_MAX));
	minDistance_memo[0][0] = 0;
	for (int i = 0; i < len1 + 1; i++) {
		for (int j = 0; j < len2 + 1; j++) {
			if (j - 1 >= 0 && i - 1 >= 0 && word1[j - 1] == word2[i - 1]) {
				minDistance_memo[i][j] = minDistance_memo[i - 1][j - 1];
				continue;
			}
			if (j - 1 >= 0) {
				minDistance_memo[i][j] = min(minDistance_memo[i][j], minDistance_memo[i][j - 1] + 1);
			}
			if (i - 1 >= 0) {
				minDistance_memo[i][j] = min(minDistance_memo[i][j], minDistance_memo[i - 1][j] + 1);
			}
			if (j - 1 >= 0 && i - 1 >= 0) {
				minDistance_memo[i][j] = min(minDistance_memo[i][j], minDistance_memo[i - 1][j - 1] + 1);
			}
		}
	}

	return minDistance_memo[len1][len2];
}

int LClabuladong::maxSubArray_dp(vector<int>& nums)
{
	int maxSubArray_memo[2];
	int res = INT_MIN;
	memset(maxSubArray_memo, -1, sizeof(maxSubArray_memo));
	for (int i = 0; i < nums.size(); i++) {
		maxSubArray_memo[1] = max(maxSubArray_memo[0] + nums[i], nums[i]);
		res = max(res, maxSubArray_memo[1]);
		maxSubArray_memo[0] = maxSubArray_memo[1];
	}

	return res;
}

int LClabuladong::maxSubArray_SlideWindow(vector<int>& nums)
{
	int left = 0, right = 0;
	int res = INT_MIN, window_sum = 0;
	for (int right = 0; right < nums.size(); right++) {
		if (window_sum < 0) {
			window_sum = 0;
			left = right;
		}
		window_sum += nums[right];
		res = max(res, window_sum);
	}
	return res;
}

int LClabuladong::maxSubArray_PrefixSum(vector<int>& nums)
{
	vector<int> pre_sum(nums.size());
	pre_sum[0] = nums[0];
	for (int i = 1; i < nums.size(); i++) {
		pre_sum[i] = pre_sum[i - 1] + nums[i];
	}
	int res = pre_sum[0], min_pre_val = min(0, pre_sum[0]);
	for (int i = 1; i < pre_sum.size(); i++) {
		res = max(res, pre_sum[i] - min_pre_val);
		min_pre_val = min(min_pre_val, pre_sum[i]);
	}

	return res;
}

int LClabuladong::longestCommonSubsequence(string text1, string text2)
{
	int len1 = text1.size(), len2 = text2.size();
	vector<vector<int>> longestCommonSubsequence_memo(2, vector<int>(len1, 0));
	int pre_max;
	for (int i = 0; i < len2; i++) {
		pre_max = 0;
		for (int j = 0; j < len1; j++) {
			if (text1[j] == text2[i]) {
				longestCommonSubsequence_memo[1][j] = max(longestCommonSubsequence_memo[0][j], pre_max + 1);
			}
			else {
				longestCommonSubsequence_memo[1][j] = longestCommonSubsequence_memo[0][j];
			}
			pre_max = max(pre_max, longestCommonSubsequence_memo[0][j]);
			longestCommonSubsequence_memo[0][j] = longestCommonSubsequence_memo[1][j];
		}
	}
	int res = 0;
	for (int i = 0; i < len1; i++) {
		res = max(res, longestCommonSubsequence_memo[0][i]);
	}

	return res;
}

int LClabuladong::minDeleteDistance(string word1, string word2)
{
	int len1 = word1.size(), len2 = word2.size();
	vector<vector<int>> minDeleteDistance_memo(2, vector<int>(len1 + 1));
	for (int i = 0; i < len1 + 1; i++) {
		minDeleteDistance_memo[0][i] = i;
	}
	for (int i = 1; i < len2 + 1; i++) {
		minDeleteDistance_memo[1][0] = i;
		for (int j = 1; j < len1 + 1; j++) {
			if (word1[j - 1] == word2[i - 1]) {
				minDeleteDistance_memo[1][j] = minDeleteDistance_memo[0][j - 1];
			}
			else {
				minDeleteDistance_memo[1][j] = min(minDeleteDistance_memo[0][j], minDeleteDistance_memo[1][j - 1]) + 1;
			}
		}
		for (int k = 0; k < len1 + 1; k++) {
			minDeleteDistance_memo[0][k] = minDeleteDistance_memo[1][k];
		}
	}
	return minDeleteDistance_memo[0][len1];
}

int LClabuladong::minimumDeleteSum(string s1, string s2)
{
	int len1 = s1.size(), len2 = s2.size();
	vector<vector<int>> minimumDeleteSum_memo(2, vector<int>(len1 + 1, 0));
	for (int i = 1; i < len1 + 1; i++) {
		minimumDeleteSum_memo[0][i] = minimumDeleteSum_memo[0][i - 1] + s1[i - 1];
	}
	for (int i = 1; i < len2 + 1; i++) {
		minimumDeleteSum_memo[1][0] = s2[i - 1] + minimumDeleteSum_memo[0][0];
		for (int j = 1; j < len1 + 1; j++) {
			if (s1[j - 1] == s2[i - 1]) {
				minimumDeleteSum_memo[1][j] = minimumDeleteSum_memo[0][j - 1];
			}
			else {
				minimumDeleteSum_memo[1][j] = min(minimumDeleteSum_memo[0][j] + s2[i - 1], minimumDeleteSum_memo[1][j - 1] + s1[j - 1]);
			}
		}
		for (int k = 0; k < len1 + 1; k++) {
			minimumDeleteSum_memo[0][k] = minimumDeleteSum_memo[1][k];
		}
	}
	return minimumDeleteSum_memo[0][len1];
}

int LClabuladong::minInsertions(string s)
{
	vector<vector<int>> minInsertions_memo;
	minInsertions_memo.resize(3, vector<int>(s.size(), 0));
	for (int i = s.size() - 1; i > 0; i--) {
		int left = 0, right = s.size() - i;
		for (int j = 0; j < i; j++) {
			if (s[left] == s[right]) {
				minInsertions_memo[2][j] = minInsertions_memo[0][j + 1];
			}
			else {
				minInsertions_memo[2][j] = min(minInsertions_memo[1][j] + 1, minInsertions_memo[1][j + 1] + 1);
			}
			++left;
			++right;
		}
		for (int k = 0; k < s.size(); k++) {
			minInsertions_memo[0][k] = minInsertions_memo[1][k];
			minInsertions_memo[1][k] = minInsertions_memo[2][k];
		}
	}
	return minInsertions_memo[2][0];
}

int LClabuladong::longestPalindromeSubseq(string s)
{
	vector<vector<int>> longestPalindromeSubseq_memo(s.size() + 1, vector<int>(s.size(), 1));
	int left, right;
	for (int i = 0; i < s.size(); i++) {
		longestPalindromeSubseq_memo[0][i] = 0;
	}
	for (int i = 2; i < s.size() + 1; i++) {
		left = 0, right = i - 1;
		for (int j = 0; j < s.size() - i + 1; j++) {
			if (s[left] != s[right]) {
				longestPalindromeSubseq_memo[i][j] = max(longestPalindromeSubseq_memo[i - 1][j], longestPalindromeSubseq_memo[i - 1][j + 1]);
			}
			else {
				longestPalindromeSubseq_memo[i][j] = longestPalindromeSubseq_memo[i - 2][j + 1] + 2;
			}
			++left;
			++right;
		}
	}
	return longestPalindromeSubseq_memo[s.size()][0];
}

bool LClabuladong::canPartition(vector<int>& nums)
{
	int sum = 0;
	for (int i : nums) {
		sum += i;
	}
	if (sum % 2 != 0) {
		return false;
	}
	int target = sum / 2;
	vector<int> canPartition_memo(target + 1, 0);
	for (int i = 0; i < nums.size(); i++) {
		for (int j = target; j >= nums[i]; j--) {
			canPartition_memo[j] = max(canPartition_memo[j], canPartition_memo[j - nums[i]] + nums[i]);
		}
	}
	return canPartition_memo[target] == target;
}

void LClabuladong::addRomanNum(int num, char arr_roman, string& res)
{
	if (num == 0) {
		return;
	}
	res += arr_roman;
	addRomanNum(num - 1, arr_roman, res);
}

string LClabuladong::intToRoman(int num)
{
	vector<int> arr_num(4, 0);
	vector<vector<char>> arr_roman(4, vector<char>(2));
	arr_roman[0][0] = 'I';
	arr_roman[0][1] = 'V';
	arr_roman[1][0] = 'X';
	arr_roman[1][1] = 'L';
	arr_roman[2][0] = 'C';
	arr_roman[2][1] = 'D';
	arr_roman[3][0] = 'M';
	string res = "";
	for (int i = 0; i < 4; i++) {
		int temp = 1000 / pow(10, i);
		arr_num[i] = num / temp;
		num -= temp * arr_num[i];
	}
	for (int i = 0; i < 4; i++) {
		if (arr_num[i] != 0) {
			if (arr_num[i] == 4 && i != 0) {
				res += arr_roman[3 - i][0];
				res += arr_roman[3 - i][1];
				continue;
			}
			if (arr_num[i] == 9 && i != 0) {
				res += arr_roman[3 - i][0];
				res += arr_roman[3 - i + 1][0];
				continue;
			}
			if (arr_num[i] >= 5) {
				res += arr_roman[3 - i][1];
				addRomanNum(arr_num[i] - 5, arr_roman[3 - i][0], res);
			}
			else {
				addRomanNum(arr_num[i], arr_roman[3 - i][0], res);
			}
		}
	}

	return res;
}

vector<vector<string>> LClabuladong::groupAnagrams(vector<string>& strs)
{
	unordered_map<string, vector<string>> m;
	vector<vector<string>> res;
	for (string& s : strs) {
		string temp_str = s;
		sort(temp_str.begin(), temp_str.end());
		m[temp_str].push_back(s);
	}
	for (pair<const string, vector<string>>& p : m) {
		res.push_back(p.second);
	}
	return res;
}

int LClabuladong::uniquePaths(int m, int n)
{
	vector<vector<int>> uniquePaths_memo(m, vector<int>(n, 1));
	for (int i = 1; i < m; i++) {
		uniquePaths_memo[i][0] = 1;
		for (int j = 1; j < n; j++) {
			uniquePaths_memo[i][j] = uniquePaths_memo[i][j - 1] + uniquePaths_memo[i - 1][j];
		}
	}
	return uniquePaths_memo[m - 1][n - 1];
}

void LClabuladong::startPartition(vector<vector<bool>>& memo, int begin, vector<string>& strs)
{
	if (begin == s_val.size()) {
		partitionPalindromic_res.push_back(strs);
		return;
	}
	for (int i = begin; i < s_val.size(); i++) {
		if (memo[begin][i]) {
			strs.push_back(s_val.substr(begin, i - begin + 1));
			startPartition(memo, i + 1, strs);
			strs.pop_back();
		}
	}
}

vector<vector<string>> LClabuladong::partitionPalindromic(string s)
{
	s_val = s;
	vector<vector<bool>> memo(s.size(), vector<bool>(s.size(), true));
	int left = 0, right = 0;
	for (int i = 1; i < s.size(); i++) {
		left = 0;
		right = i;
		for (int j = 0; j < s.size() - i; j++) {
			if (s[left] == s[right] && memo[left + 1][right - 1]) {
				memo[left][right] = true;
			}
			else {
				memo[left][right] = false;
			}
			++left;
			++right;
		}
	}
	vector<string> strs;
	for (int i = 0; i < s.size(); i++) {
		if (memo[0][i]) {
			strs.push_back(s.substr(0, i + 1));
			startPartition(memo, i + 1, strs);
			strs.pop_back();
		}
	}

	return partitionPalindromic_res;
}

int LClabuladong::findTargetSumWays(vector<int>& nums, int target)
{
	int sum = 0;
	for (int i : nums) {
		sum += i;
	}
	if (target > sum || target < -sum) {
		return 0;
	}
	vector<vector<int>> findTargetSumways_memo(nums.size(), vector<int>(sum * 2 + 1, 0));
	findTargetSumways_memo[0][nums[0] + sum] = 1;
	findTargetSumways_memo[0][sum - nums[0]] = 1;
	int left, right;
	for (int i = 1; i < nums.size(); i++) {
		for (int j = 0; j < sum * 2 + 1; j++) {
			left = j - nums[i];
			right = j + nums[i];
			if (left >= 0) {
				findTargetSumways_memo[i][j] += findTargetSumways_memo[i - 1][left];
			}
			if (right <= sum * 2) {
				findTargetSumways_memo[i][j] += findTargetSumways_memo[i - 1][right];
			}
		}
	}
	int ret = findTargetSumways_memo[nums.size() - 1][target + sum];
	if (nums[0] == 0) {
		ret *= 2;
	}
	return ret;
}

int LClabuladong::numSquares(int n)
{
	int temp_square = 1;
	int temp_i = 2;
	vector<int> nums;
	while (temp_square <= n) {
		nums.push_back(temp_square);
		temp_square = temp_i * temp_i;
		++temp_i;
	}
	vector<int> ret_arr(n + 1, 0);
	for (int i = 0; i < ret_arr.size(); i++) {
		ret_arr[i] = i;
	}
	for (int i = 1; i < nums.size(); i++) {
		for (int j = 0; j < n + 1; j++) {
			if (nums[i] <= j) {
				ret_arr[j] = min(ret_arr[j], ret_arr[j - nums[i]] + 1);
			}
		}
	}
	return ret_arr[n];
}

int LClabuladong::combinationSum4(vector<int>& nums, int target)
{
	vector<int> dp(target + 1);
	dp[0] = 1;
	for (int i = 1; i <= target; i++) {
		for (int& num : nums) {
			if (num <= i && dp[i - num] < INT_MAX - dp[i]) {
				dp[i] += dp[i - num];
			}
		}
	}
	return dp[target];
}

int LClabuladong::coinChangeII(int amount, vector<int>& coins)
{
	vector<int> coinChangeII_memo(amount + 1, 0);
	coinChangeII_memo[0] = 1;
	for (int i = 0; i < coins.size(); i++) {
		for (int j = 1; j < amount + 1; j++) {
			if (coins[i] <= j) {
				coinChangeII_memo[j] += coinChangeII_memo[j - coins[i]];
			}
		}
	}
	return coinChangeII_memo[amount];
}

int LClabuladong::minPathSum(vector<vector<int>>& grid)
{
	int width, height;
	height = grid.size();
	width = grid[0].size();
	vector<int> minPathSum_memo(width, 0);
	minPathSum_memo[0] = grid[0][0];
	for (int i = 1; i < width; i++) {
		minPathSum_memo[i] = minPathSum_memo[i - 1] + grid[0][i];
	}
	for (int i = 1; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j == 0) {
				minPathSum_memo[j] += grid[i][0];
				continue;
			}
			minPathSum_memo[j] = min(minPathSum_memo[j - 1], minPathSum_memo[j]) + grid[i][j];
		}
	}
	return minPathSum_memo[width - 1];
}

int LClabuladong::calculateMinimumHP(vector<vector<int>>& dungeon)
{
	int m = dungeon.size(), n = dungeon[0].size();
	vector<vector<int>> dp(m, vector<int>(n));
	//base case
	dp[m - 1][n - 1] = max(1, 1 - dungeon[m - 1][n - 1]);
	for (int i = m - 2; i >= 0; --i) {
		dp[i][n - 1] = max(dp[i + 1][n - 1] - dungeon[i][n - 1], 1);
	}
	for (int i = n - 2; i >= 0; --i) {
		dp[m - 1][i] = max(dp[m - 1][i + 1] - dungeon[m - 1][i], 1);
	}

	for (int i = m - 2; i >= 0; --i) {
		for (int j = n - 2; j >= 0; --j) {
			dp[i][j] = min(dp[i + 1][j] - dungeon[i][j],
				dp[i][j + 1] - dungeon[i][j]);
			if (dp[i][j] <= 0) dp[i][j] = 1;
		}
	}
	return dp[0][0];
}

int LClabuladong::findRotateSteps(string ring, string key)
{
	findRotateSteps_memo.resize(ring.size(), vector<int>(key.size(), 0));
	for (int i = 0; i < ring.size(); i++) {
		int temp_char = ring[i];
		findRotateSteps_char_to_index[temp_char].push_back(i);
	}
	return findRotateSteps_dp(ring, 0, key, 0);
}

int LClabuladong::findRotateSteps_dp(string ring, int i, string key, int j)
{
	if (j == key.size()) {
		return 0;
	}
	if (findRotateSteps_memo[i][j] != 0) {
		return findRotateSteps_memo[i][j];
	}
	int delta;
	char key_char = key[j];
	int ret = INT_MAX;
	for (int index : findRotateSteps_char_to_index[key_char]) {
		delta = abs(index - i);
		delta = min(delta, int(ring.size()) - delta);
		int latter_delta = findRotateSteps_dp(ring, index, key, j + 1);
		ret = min(ret, latter_delta + delta + 1);
	}
	findRotateSteps_memo[i][j] = ret;
	return ret;
}

vector<string> LClabuladong::letterCombinations(string digits)
{
	if (digits.size() == 0) {
		return letterCombinations_ret;
	}
	letterCombinations_digit_to_letter[0] = "";
	char j = 'a';
	for (int i = 2; i < 10; i++) {
		letterCombinations_digit_to_letter[i].push_back(j++);
		letterCombinations_digit_to_letter[i].push_back(j++);
		letterCombinations_digit_to_letter[i].push_back(j++);
		if (i == 7 || i == 9) {
			letterCombinations_digit_to_letter[i].push_back(j++);
		}
	}
	string string_res;
	digits2letters(digits, 0, string_res);

	return letterCombinations_ret;
}

void LClabuladong::digits2letters(string digits, int begin, string res)
{
	if (begin == digits.size()) {
		letterCombinations_ret.push_back(res);
	}
	for (int i = 0; i < letterCombinations_digit_to_letter[digits[begin] - '0'].size(); i++) {
		res += letterCombinations_digit_to_letter[digits[begin] - '0'][i];
		digits2letters(digits, begin + 1, res);
		res.pop_back();
	}
}

ListNode* LClabuladong::swapPairs(ListNode* head)
{
	if (head == NULL) {
		return NULL;
	}
	ListNode* node_ptr1, * node_ptr2, * node_ptr3 = NULL;
	node_ptr1 = head;
	node_ptr2 = head->next;
	ListNode* temp_head = new ListNode();
	temp_head->next = head;
	node_ptr3 = temp_head;
	while (node_ptr2 != NULL && node_ptr1 != NULL) {
		node_ptr1->next = node_ptr2->next;
		node_ptr2->next = node_ptr1;
		node_ptr3->next = node_ptr2;
		node_ptr3 = node_ptr1;
		node_ptr1 = node_ptr1->next;
		if (node_ptr1 == NULL) break;
		node_ptr2 = node_ptr1->next;
	}
	head = temp_head->next;
	return head;
}

void LClabuladong::recoverTree(TreeNode* root)
{
	vector<TreeNode*> middle_order;
	recoverTree_dfs(root, middle_order);
	int temp;
	TreeNode* temp_node1 = NULL;
	TreeNode* temp_node2 = NULL;
	if (recoverTree_res.size() == 2) {
		temp_node1 = middle_order[recoverTree_res[0]];
		temp_node2 = middle_order[recoverTree_res[1] + 1];
	}
	else if (recoverTree_res.size() == 1) {
		temp_node1 = middle_order[recoverTree_res[0]];
		temp_node2 = middle_order[recoverTree_res[0] + 1];
	}
	temp = temp_node1->val;
	temp_node1->val = temp_node2->val;
	temp_node2->val = temp;
}

void LClabuladong::recoverTree_dfs(TreeNode* root, vector<TreeNode*>& order)
{
	if (root->left != NULL) {
		recoverTree_dfs(root->left, order);
	}
	if (order.size() != 0 && root->val < order.back()->val) {
		recoverTree_res.push_back(order.size() - 1);
	}
	order.push_back(root);
	if (root->right != NULL) {
		recoverTree_dfs(root->right, order);
	}
}

vector<vector<string>> LClabuladong::solveNQueens(int n)
{
	vector<string> chessboard(n, string(n, '.'));
	solveNQueens_recursion(chessboard, 0);
	return solveNQueens_res;
}

void LClabuladong::solveNQueens_recursion(vector<string>& chessboard, int queen_order)
{
	if (queen_order == chessboard.size()) {
		solveNQueens_res.push_back(chessboard);
		return;
	}
	for (int i = 0; i < chessboard.size(); i++) {
		if (!solveNQueens_isAvailable(chessboard, queen_order, i)) {
			continue;
		}
		chessboard[queen_order][i] = 'Q';
		solveNQueens_recursion(chessboard, queen_order + 1);
		chessboard[queen_order][i] = '.';
	}
}

bool LClabuladong::solveNQueens_isAvailable(vector<string>& chessboard, int row, int col)
{
	for (int i = 0; i < row; i++) {
		if (chessboard[i][col] == 'Q') {
			return false;
		}
	}
	for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
		if (chessboard[i][j] == 'Q') {
			return false;
		}
	}
	for (int i = row - 1, j = col + 1; i >= 0 && j < chessboard.size(); i--, j++) {
		if (chessboard[i][j] == 'Q') {
			return false;
		}
	}
	return true;
}

bool LClabuladong::canPartitionKSubsets(vector<int>& nums, int k)
{
	int sum = 0;
	for (int i = 0; i < nums.size(); i++) sum += nums[i];
	if (sum % k != 0) return false;
	int target = sum / k;
	// 排序优化
	sort(nums.begin(), nums.end(), greater<int>());
	vector<int> buckets(k, target);
	return canPartitionKSubsets_loadBuckets(buckets, k, nums, 0);
}

bool LClabuladong::canPartitionKSubsets_loadBuckets(vector<int>& buckets, int bucket_index, vector<int>& nums, int nums_index)
{
	// 结束条件优化
	if (nums_index == nums.size()) return true;
	for (int i = 0; i < bucket_index; i++) {
		// 优化点二
		if (i > 0 && buckets[i] == buckets[i - 1]) continue;
		// 剪枝
		if (buckets[i] - nums[nums_index] < 0) continue;
		buckets[i] -= nums[nums_index];
		if (canPartitionKSubsets_loadBuckets(buckets, bucket_index, nums, nums_index + 1)) return true;
		buckets[i] += nums[nums_index];
	}
	return false;
}

int LClabuladong::numIslands(vector<vector<char>>& grid)
{
	int width, height;
	int ret = 0;
	height = grid.size();
	width = grid[0].size();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (grid[i][j] == '1') {
				numIslands_connect(grid, i, j);
				++ret;
			}
		}
	}
	return ret;
}

void LClabuladong::numIslands_connect(vector<vector<char>>& grid, int i, int j)
{
	grid[i][j] = '-';
	if (j + 1 < grid[0].size() && grid[i][j + 1] == '1') {
		numIslands_connect(grid, i, j + 1);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == '1') {
		numIslands_connect(grid, i + 1, j);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == '1') {
		numIslands_connect(grid, i - 1, j);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == '1') {
		numIslands_connect(grid, i, j - 1);
	}
}

int LClabuladong::closedIsland(vector<vector<int>>& grid)
{
	int ret = 0;
	int height = grid.size(), width = grid[0].size();
	for (int i = 0; i < width; i++) {
		if (grid[0][i] == 0) {
			closedIsland_reverse(grid, 0, i);
		}
		if (grid[height - 1][i] == 0) {
			closedIsland_reverse(grid, height - 1, i);
		}
	}
	for (int i = 0; i < height; i++) {
		if (grid[i][0] == 0) {
			closedIsland_reverse(grid, i, 0);
		}
		if (grid[i][width - 1] == 0) {
			closedIsland_reverse(grid, i, width - 1);
		}
	}

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (grid[i][j] == 0) {
				closedIsland_isClosedIsland(grid, i, j);
				++ret;
			}
		}
	}
	return ret;
}

void LClabuladong::closedIsland_isClosedIsland(vector<vector<int>>& grid, int i, int j)
{
	grid[i][j] = 2;
	if (j + 1 < grid[0].size() && grid[i][j + 1] == 0) {
		closedIsland_isClosedIsland(grid, i, j + 1);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == 0) {
		closedIsland_isClosedIsland(grid, i + 1, j);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == 0) {
		closedIsland_isClosedIsland(grid, i - 1, j);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == 0) {
		closedIsland_isClosedIsland(grid, i, j - 1);
	}
}

void LClabuladong::closedIsland_reverse(vector<vector<int>>& grid, int i, int j)
{
	grid[i][j] = 3;
	if (j + 1 < grid[0].size() && grid[i][j + 1] == 0) {
		closedIsland_reverse(grid, i, j + 1);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == 0) {
		closedIsland_reverse(grid, i + 1, j);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == 0) {
		closedIsland_reverse(grid, i - 1, j);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == 0) {
		closedIsland_reverse(grid, i, j - 1);
	}
}

int LClabuladong::numEnclaves(vector<vector<int>>& grid)
{
	int ret = 0;
	int height = grid.size(), width = grid[0].size();
	for (int i = 0; i < width; i++) {
		if (grid[0][i] == 1) {
			numEnclaves_reverse(grid, 0, i);
		}
		if (grid[height - 1][i] == 1) {
			numEnclaves_reverse(grid, height - 1, i);
		}
	}
	for (int i = 0; i < height; i++) {
		if (grid[i][0] == 1) {
			numEnclaves_reverse(grid, i, 0);
		}
		if (grid[i][width - 1] == 1) {
			numEnclaves_reverse(grid, i, width - 1);
		}
	}
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (grid[i][j] == 1) {
				numEnclaves_isClosed(grid, i, j, ret);
			}
		}
	}

	return ret;
}

void LClabuladong::numEnclaves_isClosed(vector<vector<int>>& grid, int i, int j, int& ret)
{
	grid[i][j] = 2;
	++ret;
	if (j + 1 < grid[0].size() && grid[i][j + 1] == 1) {
		numEnclaves_isClosed(grid, i, j + 1, ret);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == 1) {
		numEnclaves_isClosed(grid, i + 1, j, ret);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == 1) {
		numEnclaves_isClosed(grid, i - 1, j, ret);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == 1) {
		numEnclaves_isClosed(grid, i, j - 1, ret);
	}
}

void LClabuladong::numEnclaves_reverse(vector<vector<int>>& grid, int i, int j)
{
	grid[i][j] = 3;
	if (j + 1 < grid[0].size() && grid[i][j + 1] == 1) {
		numEnclaves_reverse(grid, i, j + 1);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == 1) {
		numEnclaves_reverse(grid, i + 1, j);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == 1) {
		numEnclaves_reverse(grid, i - 1, j);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == 1) {
		numEnclaves_reverse(grid, i, j - 1);
	}
}

int LClabuladong::maxAreaOfIsland(vector<vector<int>>& grid)
{
	int width, height;
	height = grid.size();
	width = grid[0].size();
	int maxAreaOfIsland_max_area = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (grid[i][j] == 1) {
				maxAreaOfIsland_max_area = max(maxAreaOfIsland_max_area, maxAreaOfIsland_countArea(grid, i, j));
			}
		}
	}
	return maxAreaOfIsland_max_area;
}

int LClabuladong::maxAreaOfIsland_countArea(vector<vector<int>>& grid, int i, int j)
{
	grid[i][j] = 2;
	int area = 1;
	if (j + 1 < grid[0].size() && grid[i][j + 1] == 1) {
		area += maxAreaOfIsland_countArea(grid, i, j + 1);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == 1) {
		area += maxAreaOfIsland_countArea(grid, i + 1, j);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == 1) {
		area += maxAreaOfIsland_countArea(grid, i - 1, j);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == 1) {
		area += maxAreaOfIsland_countArea(grid, i, j - 1);
	}
	return area;
}

int LClabuladong::countSubIslands(vector<vector<int>>& grid1, vector<vector<int>>& grid2)
{
	int ret = 0;
	int height, width;
	height = grid1.size();
	width = grid1[0].size();
	//找出grid1的岛屿，并叠图
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (grid1[i][j] == 1) {
				countSubIslands_islandReverse(grid1, i, j, 1, 2);
			}
			grid2[i][j] += grid1[i][j];
			if (grid2[i][j] == 2) {
				grid2[i][j] = 0;
			}
		}
	}
	//将grid2中不符合子岛屿要求的岛屿置0（与1相邻的都是不符合要求的）
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (grid2[i][j] == 1) {
				countSubIslands_islandReverse(grid2, i, j, 3, 0);
			}
		}
	}
	//grid2中剩下的岛屿全是grid1的子岛屿
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (grid2[i][j] == 3) {
				++ret;
				countSubIslands_islandReverse(grid2, i, j, 3, 2);
			}
		}
	}

	return ret;
}

void LClabuladong::countSubIslands_islandReverse(vector<vector<int>>& grid, int i, int j, int pave, int val)
{
	grid[i][j] = val;
	if (j + 1 < grid[0].size() && grid[i][j + 1] == pave) {
		countSubIslands_islandReverse(grid, i, j + 1, pave, val);
	}
	if (i + 1 < grid.size() && grid[i + 1][j] == pave) {
		countSubIslands_islandReverse(grid, i + 1, j, pave, val);
	}
	if (i - 1 >= 0 && grid[i - 1][j] == pave) {
		countSubIslands_islandReverse(grid, i - 1, j, pave, val);
	}
	if (j - 1 >= 0 && grid[i][j - 1] == pave) {
		countSubIslands_islandReverse(grid, i, j - 1, pave, val);
	}
}

int LClabuladong::minDepth(TreeNode* root)
{
	if (!root) {
		return 0;
	}
	queue<TreeNode*> q_nodes;
	TreeNode* temp_node = NULL;
	q_nodes.push(root);
	int ret = 0;
	while (!q_nodes.empty()) {
		++ret;
		int q_size = q_nodes.size();
		for (int i = 0; i < q_size; i++) {
			temp_node = q_nodes.front();
			q_nodes.pop();
			if (temp_node->left == NULL && temp_node->right == NULL) {
				return ret;
			}
			if (temp_node->right) {
				q_nodes.push(temp_node->right);
			}
			if (temp_node->left) {
				q_nodes.push(temp_node->left);
			}
		}
	}
	return ret;
}

int LClabuladong::openLock(vector<string>& deadends, string target)
{
	if (target == "0000") {
		return 0;
	}
	queue<string> q_lock;
	bool visited[10][10][10][10] = { false };
	int ret = 0;
	string temp_string;
	int q_size;
	q_lock.push("0000");
	for (int i = 0; i < deadends.size(); i++) {
		visited[deadends[i][0] - '0'][deadends[i][1] - '0'][deadends[i][2] - '0'][deadends[i][3] - '0'] = true;
	}
	if (visited[0][0][0][0]) {
		return -1;
	}

	while (!q_lock.empty()) {
		q_size = q_lock.size();
		for (int i = 0; i < q_size; i++) {
			temp_string = q_lock.front();
			q_lock.pop();
			if (temp_string == target) {
				return ret;
			}
			if (visited[temp_string[0] - '0'][temp_string[1] - '0'][temp_string[2] - '0'][temp_string[3] - '0']) {
				continue;
			}
			for (int i = 0; i < 4; i++) {
				string temp = temp_string;
				if (temp[i] == '9') {
					temp[i] = '0';
				}
				else {
					temp[i] += 1;
				}
				q_lock.push(temp);
				temp = temp_string;
				if (temp[i] == '0') {
					temp[i] = '9';
				}
				else {
					temp[i] -= 1;
				}
				q_lock.push(temp);
			}
			visited[temp_string[0] - '0'][temp_string[1] - '0'][temp_string[2] - '0'][temp_string[3] - '0'] = true;
		}
		++ret;
	}

	return -1;
}

int LClabuladong::slidingPuzzle(vector<vector<int>>& board)
{
	queue<vector<vector<int>>> q_boards;
	bool visited[6][6][6][6][6][6] = { false };
	q_boards.push(board);
	int ret = 0;
	if (slidingPuzzle_isOK(board)) {
		return ret;
	}
	while (!q_boards.empty()) {
		int q_size = q_boards.size();
		for (int i = 0; i < q_size; i++) {
			vector<vector<int>> temp_board = q_boards.front();
			q_boards.pop();
			pair<int, int> zero_index = slidingPuzzle_findZero(temp_board);
			int x = zero_index.first, y = zero_index.second;
			vector<vector<int>> temp = temp_board;
			if (x + 1 < 2) {
				slidingPuzzle_swapTwoSquare(temp, x, y, x + 1, y);
				if (slidingPuzzle_isOK(temp)) {
					return ret + 1;
				}
				else {
					if (!slidingPuzzle_isVisited(temp, visited))
					{
						q_boards.push(temp);
					}
				}
			}
			temp = temp_board;
			if (x - 1 >= 0) {
				slidingPuzzle_swapTwoSquare(temp, x, y, x - 1, y);
				if (slidingPuzzle_isOK(temp)) {
					return ret + 1;
				}
				else {
					if (!slidingPuzzle_isVisited(temp, visited))
					{
						q_boards.push(temp);
					}
				}
			}
			temp = temp_board;
			if (y + 1 < 3) {
				slidingPuzzle_swapTwoSquare(temp, x, y, x, y + 1);
				if (slidingPuzzle_isOK(temp)) {
					return ret + 1;
				}
				else {
					if (!slidingPuzzle_isVisited(temp, visited))
					{
						q_boards.push(temp);
					}
				}
			}
			temp = temp_board;
			if (y - 1 >= 0) {
				slidingPuzzle_swapTwoSquare(temp, x, y, x, y - 1);
				if (slidingPuzzle_isOK(temp)) {
					return ret + 1;
				}
				else {
					if (!slidingPuzzle_isVisited(temp, visited))
					{
						q_boards.push(temp);
					}
				}
			}
			slidingPuzzle_visit(temp_board, visited);
		}
		++ret;
	}
	return -1;
}

bool LClabuladong::slidingPuzzle_isOK(vector<vector<int>>& board)
{
	if (board[0][0] == 1 && board[0][1] == 2 && board[0][2] == 3 &&
		board[1][0] == 4 && board[1][1] == 5 && board[1][2] == 0) {
		return true;
	}
	return false;
}

void LClabuladong::slidingPuzzle_swapTwoSquare(vector<vector<int>>& board, int x1, int y1, int x2, int y2)
{
	int temp = board[x1][y1];
	board[x1][y1] = board[x2][y2];
	board[x2][y2] = temp;
}

pair<int, int> LClabuladong::slidingPuzzle_findZero(vector<vector<int>>& board)
{
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			if (board[i][j] == 0) {
				return pair<int, int>(i, j);
			}
		}
	}
	return pair<int, int>();
}

bool LClabuladong::slidingPuzzle_isVisited(vector<vector<int>>& board, bool visited[6][6][6][6][6][6])
{
	return visited[board[0][0]][board[0][1]][board[0][2]][board[1][0]][board[1][1]][board[1][2]];
}

void LClabuladong::slidingPuzzle_visit(vector<vector<int>>& board, bool visited[6][6][6][6][6][6])
{
	visited[board[0][0]][board[0][1]][board[0][2]][board[1][0]][board[1][1]][board[1][2]] = true;
}

int LClabuladong::findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k)
{
	vector<vector<int>> price_table = findCheapestPrice_creatPriceTable(flights, n);
	vector<vector<int>> dp(2, vector<int>(n, INT_MAX));
	dp[0][dst] = k;
	dp[1][dst] = 0;
	queue<int> myq;
	myq.push(dst);
	while (!myq.empty()) {
		int q_size = myq.size();
		for (int i = 0; i < q_size; i++) {
			int start = myq.front();
			myq.pop();
			for (int i = 0; i < n; i++) {
				if (price_table[start][i] > 0) {
					int new_d = dp[0][start] - 1;
					int new_price = dp[1][start] + price_table[start][i];
					if (new_d == -1 && i != src) {
						continue;
					}
					if (new_price < dp[1][i]) {
						dp[0][i] = new_d;
						dp[1][i] = new_price;
						myq.push(i);
					}
				}
			}
		}
	}
	if (dp[0][src] == INT_MAX) {
		return -1;
	}
	return dp[1][src];
}

vector<vector<int>> LClabuladong::findCheapestPrice_creatPriceTable(vector<vector<int>>& flights, int n)
{
	vector<vector<int>> table(n, vector<int>(n, -1));
	for (int i = 0; i < n; i++) {
		table[i][i] = 0;
	}
	for (int i = 0; i < flights.size(); i++) {
		table[flights[i][1]][flights[i][0]] = flights[i][2];
	}
	return table;
}

bool LClabuladong::PredictTheWinner(vector<int>& nums)
{
	int length = nums.size();
	auto dp = vector<vector<int>>(length, vector<int>(length));
	for (int i = 0; i < length; i++) {
		dp[i][i] = nums[i];
	}
	for (int i = length - 2; i >= 0; i--) {
		for (int j = i + 1; j < length; j++) {
			dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
		}
	}
	return dp[0][length - 1] >= 0;
}

string LClabuladong::getHint(string secret, string guess)
{
	int bull_nums = 0, cow_nums = 0;
	unordered_map<int, int> guess_char_num, secret_char_num;
	for (int i = 0; i < guess.size(); i++) {
		if (secret[i] == guess[i]) {
			++bull_nums;
		}
		else {
			guess_char_num[guess[i] - '0'] += 1;
			secret_char_num[secret[i] - '0'] += 1;
		}
	}
	for (pair<int, int> iter : secret_char_num) {
		cow_nums += min(iter.second, guess_char_num[iter.first]);
	}
	return string(to_string(bull_nums) + 'A' + to_string(cow_nums) + 'B');
}

vector<int> LClabuladong::productExceptSelf(vector<int>& nums)
{
	vector<int> pre_product_arr(nums.size() + 1, 1), suf_product_arr(nums.size() + 1, 1);
	for (int i = 1; i < pre_product_arr.size(); i++) {
		pre_product_arr[i] = pre_product_arr[i - 1] * nums[i - 1];
	}
	for (int i = suf_product_arr.size() - 2; i >= 0; i--) {
		suf_product_arr[i] = suf_product_arr[i + 1] * nums[i];
	}
	vector<int> ret(nums.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = pre_product_arr[i] * suf_product_arr[i + 1];
	}
	return ret;
}

int LClabuladong::countGoodNumbers(long long n)
{
	int mod = 1000000007;
	// 快速幂求出 x^y % mod
	auto quickmul = [mod](int x, long long y) -> int {
		int ret = 1, mul = x;
		while (y > 0) {
			if (y % 2 == 1) {
				ret = (long long)ret * mul % mod;
			}
			mul = (long long)mul * mul % mod;
			y /= 2;
		}
		return ret;
		};

	return (long long)quickmul(5, (n + 1) / 2) * quickmul(4, n / 2) % mod;
}

bool LClabuladong::wordBreak(string s, vector<string>& wordDict)
{
	unordered_set<string> myset;
	for (int i = 0; i < wordDict.size(); i++) {
		myset.insert(wordDict[i]);
	}
	vector<bool> dp(s.size() + 1, false);
	dp[0] = true;
	for (int i = 1; i < dp.size(); i++) {
		for (int j = 0; j < i; j++) {
			if (dp[j] && myset.count(s.substr(j, i - j))) {
				dp[i] = true;
				break;
			}
		}
	}
	return dp[dp.size() - 1];
}

string LClabuladong::multiply(string num1, string num2)
{
	vector<vector<int>> cumulative(num1.size());
	string ret;
	int ptr1 = num1.size() - 1, ptr2 = num2.size() - 1;
	int cur;
	int carry = 0;

	for (ptr1; ptr1 >= 0; ptr1--) {
		carry = 0;
		for (int i = 0; i < num1.size() - 1 - ptr1; i++) {
			cumulative[num1.size() - 1 - ptr1].push_back(0);
		}
		for (ptr2 = num2.size() - 1; ptr2 >= -1; ptr2--) {
			if (ptr2 == -1) {
				if (carry == 0) {
					break;
				}
				cumulative[num1.size() - 1 - ptr1].push_back(carry);
				break;
			}
			cur = int(num1[ptr1] - '0') * int(num2[ptr2] - '0') + carry;
			carry = cur / 10;
			cur = cur % 10;
			cumulative[num1.size() - 1 - ptr1].push_back(cur);
		}
	}

	vector<int> v_ret = cumulative[0];
	for (int i = 1; i < cumulative.size(); i++) {
		v_ret = multiply_add(v_ret, cumulative[i]);
	}
	int ret_begin = v_ret.size() - 1;
	for (int i = v_ret.size() - 1; i >= 0; i--) {
		if (i == 0) {
			ret_begin = 0;
			break;
		}
		if (v_ret[i] == 0) {
			continue;
		}
		else {
			ret_begin = i;
			break;
		}
	}
	for (int i = ret_begin; i >= 0; i--) {
		ret += char(v_ret[i] + int('0'));
	}

	return ret;
}

vector<int> LClabuladong::multiply_add(vector<int> num1, vector<int> num2)
{
	vector<int> ret;
	int i = 0, j = 0;
	int cur = 0, carry = 0;
	while (i < num1.size() && j < num2.size()) {
		cur = num1[i] + num2[j] + carry;
		carry = cur / 10;
		cur = cur % 10;
		ret.push_back(cur);
		++i;
		++j;
	}
	while (i < num1.size()) {
		cur = num1[i] + carry;
		carry = cur / 10;
		cur = cur % 10;
		ret.push_back(cur);
		++i;
	}
	while (j < num2.size()) {
		cur = num2[j] + carry;
		carry = cur / 10;
		cur = cur % 10;
		ret.push_back(cur);
		++j;
	}
	if (carry) {
		ret.push_back(carry);
	}
	return ret;
}

int LClabuladong::findNthDigit(int n)
{
	if (n < 10) {
		return n;
	}
	long long n_ = n;
	long long base = 0;
	long long ratio = 1;
	while (n_ > 0) {
		base++;
		n_ = n_ - base * 9 * ratio;
		ratio *= 10;
	}
	ratio /= 10;
	n_ = n_ + base * 9 * ratio;
	int index = (n_ - 1) / base;
	index = index + ratio;
	return int(to_string(index)[(n_ - 1) % base] - '0');
}

int LClabuladong::lastRemaining(int n)
{
	bool left_to_right = true;
	int arr_amount = n;
	int a0 = 1, d = 1;
	while (arr_amount != 1) {
		if (arr_amount % 2 != 0) {
			a0 = a0 + d;
		}
		else {
			if (left_to_right) {
				a0 = a0 + d;
			}
			else {
				a0 = a0;
			}
		}
		arr_amount /= 2;
		left_to_right = !left_to_right;
		d *= 2;
	}
	return a0;
}

bool LClabuladong::escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target)
{
	vector<int> source(2);
	int distance = escapeGhosts_distance(source, target);
	for (auto& ghost : ghosts) {
		int ghostDistance = escapeGhosts_distance(ghost, target);
		if (ghostDistance <= distance) {
			return false;
		}
	}
	return true;
}

int LClabuladong::escapeGhosts_distance(vector<int>& source, vector<int>& target)
{
	return abs(source[0] - target[0]) + abs(source[1] - target[1]);
}

int LClabuladong::lenLongestFibSubseq(vector<int>& arr)
{
	unordered_map<int, int> indices;
	for (int i = 0; i < arr.size(); i++) {
		indices[arr[i]] = i;
	}
	int n = arr.size();
	vector<vector<int>> lenLongestFibSubseq_dp(n, vector<int>(n, 2));
	int ret = 0;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			int diff = arr[j] - arr[i];
			if (indices.count(diff)) {
				int index = indices[diff];
				if (index < i) {
					lenLongestFibSubseq_dp[i][j] = max(lenLongestFibSubseq_dp[i][j], lenLongestFibSubseq_dp[index][i] + 1);
				}
			}
			ret = max(ret, lenLongestFibSubseq_dp[i][j]);
		}
	}
	return ret > 2 ? ret : 0;;
}

int LClabuladong::maxSumDivThree(vector<int>& nums)
{
	int n = nums.size();
	vector<vector<int>> maxSumDivThree_dp(n + 1, vector<int>(3, 0));
	maxSumDivThree_dp[0][1] = INT_MIN, maxSumDivThree_dp[0][2] = INT_MIN;
	for (int i = 1; i <= n; i++) {
		if (nums[i - 1] % 3 == 0) {
			maxSumDivThree_dp[i][0] = max(maxSumDivThree_dp[i - 1][0], maxSumDivThree_dp[i - 1][0] + nums[i - 1]);
			maxSumDivThree_dp[i][1] = max(maxSumDivThree_dp[i - 1][1], maxSumDivThree_dp[i - 1][1] + nums[i - 1]);
			maxSumDivThree_dp[i][2] = max(maxSumDivThree_dp[i - 1][2], maxSumDivThree_dp[i - 1][2] + nums[i - 1]);
		}
		else if (nums[i - 1] % 3 == 1) {
			maxSumDivThree_dp[i][0] = max(maxSumDivThree_dp[i - 1][0], maxSumDivThree_dp[i - 1][2] + nums[i - 1]);
			maxSumDivThree_dp[i][1] = max(maxSumDivThree_dp[i - 1][1], maxSumDivThree_dp[i - 1][0] + nums[i - 1]);
			maxSumDivThree_dp[i][2] = max(maxSumDivThree_dp[i - 1][2], maxSumDivThree_dp[i - 1][1] + nums[i - 1]);
		}
		else if (nums[i - 1] % 3 == 2) {
			maxSumDivThree_dp[i][0] = max(maxSumDivThree_dp[i - 1][0], maxSumDivThree_dp[i - 1][1] + nums[i - 1]);
			maxSumDivThree_dp[i][1] = max(maxSumDivThree_dp[i - 1][1], maxSumDivThree_dp[i - 1][2] + nums[i - 1]);
			maxSumDivThree_dp[i][2] = max(maxSumDivThree_dp[i - 1][2], maxSumDivThree_dp[i - 1][0] + nums[i - 1]);
		}
	}
	return maxSumDivThree_dp[n][0];
}

double LClabuladong::knightProbability(int n, int k, int row, int column)
{
	vector<vector<vector<double>>> dp(k + 1, vector<vector<double>>(n, vector<double>(n, 0)));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			dp[0][i][j] = 1;
		}
	}
	for (int p = 1; p <= k; p++) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				dp[p][i][j] = knightProbability_getProbability(n, i, j, dp[p - 1]);
			}
		}
	}
	return dp[k][row][column];
}

double LClabuladong::knightProbability_getProbability(int n, int x, int y, vector<vector<double>>& dp)
{
	double probability = 0;
	if (x - 2 >= 0) {
		if (y - 1 >= 0) {
			probability += (dp[x - 2][y - 1] * 0.125);
		}
		if (y + 1 < n) {
			probability += (dp[x - 2][y + 1] * 0.125);
		}
	}
	if (x + 2 < n) {
		if (y - 1 >= 0) {
			probability += (dp[x + 2][y - 1] * 0.125);
		}
		if (y + 1 < n) {
			probability += (dp[x + 2][y + 1] * 0.125);
		}
	}
	if (y + 2 < n) {
		if (x - 1 >= 0) {
			probability += (dp[x - 1][y + 2] * 0.125);
		}
		if (x + 1 < n) {
			probability += (dp[x + 1][y + 2] * 0.125);
		}
	}
	if (y - 2 >= 0) {
		if (x - 1 >= 0) {
			probability += (dp[x - 1][y - 2] * 0.125);
		}
		if (x + 1 < n) {
			probability += (dp[x + 1][y - 2] * 0.125);
		}
	}
	return probability;
}

vector<int> LClabuladong::numsSameConsecDiff(int n, int k)
{
	vector<int> cur_num(n, 0);
	for (int i = 1; i < 10; i++) {
		cur_num[0] = i;
		numsSameConsecDiff_recursion(n, k, 0, cur_num);
	}
	return numsSameConsecDiff_ret;
}

void LClabuladong::numsSameConsecDiff_recursion(int n, int k, int p, vector<int>& cur)
{
	if (n - 1 == p) {
		int add_num = 0;
		for (int i = 0; i < n; i++) {
			add_num = add_num * 10 + cur[i];
		}
		numsSameConsecDiff_ret.push_back(add_num);
		return;
	}
	int temp = cur[p];
	if (temp + k < 10) {
		cur[p + 1] = temp + k;
		numsSameConsecDiff_recursion(n, k, p + 1, cur);
	}
	if (k != 0 && temp - k >= 0) {
		cur[p + 1] = temp - k;
		numsSameConsecDiff_recursion(n, k, p + 1, cur);
	}
	return;
}

bool LClabuladong::isMatch(string s, string p)
{
	int row = s.size(), col = p.size();
	vector<vector<int>> isMatch_dp(row + 1, vector<int>(col + 1, false));
	isMatch_dp[0][0] = true;
	for (int i = 0; i <= row; i++) {
		for (int j = 1; j <= col; j++) {
			if (p[j - 1] == '*') {
				isMatch_dp[i][j] = isMatch_dp[i][j - 2];
				if (i != 0 && (p[j - 2] == s[i - 1] || p[j - 2] == '.')) {
					isMatch_dp[i][j] = isMatch_dp[i - 1][j] ? true : isMatch_dp[i][j];
				}
			}
			else {
				if (i != 0 && (s[i - 1] == p[j - 1] || p[j - 1] == '.')) {
					isMatch_dp[i][j] = isMatch_dp[i - 1][j - 1];
				}
			}
		}
	}
	return isMatch_dp[row][col];
}

int LClabuladong::superEggDrop(int k, int n)
{
	return superEggDrop_dp(k, n);
}

int LClabuladong::superEggDrop_dp(int k, int n)
{
	if (superEggDrop_memo.find(n * 100 + k) == superEggDrop_memo.end()) {
		int ans;
		if (n == 0) {
			ans = 0;
		}
		else if (k == 1) {
			ans = n;
		}
		else {
			int lo = 1, hi = n;
			while (lo + 1 < hi) {
				int x = (lo + hi) / 2;
				int t1 = superEggDrop_dp(k - 1, x - 1);
				int t2 = superEggDrop_dp(k, n - x);

				if (t1 < t2) {
					lo = x;
				}
				else if (t1 > t2) {
					hi = x;
				}
				else {
					lo = hi = x;
				}
			}

			ans = 1 + min(max(superEggDrop_dp(k - 1, lo - 1), superEggDrop_dp(k, n - lo)),
				max(superEggDrop_dp(k - 1, hi - 1), superEggDrop_dp(k, n - hi)));
		}

		superEggDrop_memo[n * 100 + k] = ans;
	}

	return superEggDrop_memo[n * 100 + k];
}

bool LClabuladong::stoneGame(vector<int>& piles)
{
	int n = piles.size();
	vector<vector<int>> stoneGame_dp(n, vector<int>(n, 0));
	for (int i = 0; i < n; i++) {
		stoneGame_dp[0][i] = piles[i];
	}
	for (int i = 1; i < n; i++) {
		int x = 0;
		int y = i;
		for (int j = 0; j < n - i; j++) {
			stoneGame_dp[i][j] = max(piles[y] - stoneGame_dp[i - 1][j], piles[x] - stoneGame_dp[i - 1][j + 1]);
			++y;
			++x;
		}
	}
	return stoneGame_dp[n - 1][0] > 0;
}

int LClabuladong::rob(vector<int>& nums)
{
	int n = nums.size();
	vector<vector<int>> rob_dp(n, vector<int>(n, 0));
	for (int i = 0; i < n; i++) {
		rob_dp[0][i] = nums[i];
	}
	for (int i = 1; i < n; i++) {
		int x = 0;
		int y = i;
		for (int j = 0; j < n - i; j++) {
			if (i - 2 < 0) {
				rob_dp[i][j] = max(max(rob_dp[i - 1][j], rob_dp[i - 1][j + 1]), max(nums[y], nums[x]));
			}
			else {
				rob_dp[i][j] = max(max(rob_dp[i - 1][j], rob_dp[i - 1][j + 1]), max(nums[y] + rob_dp[i - 2][j], nums[x] + rob_dp[i - 2][j + 2]));
			}
			++x;
			++y;
		}
	}
	return rob_dp[n - 1][0];
}

int LClabuladong::MachineHandling(int n, vector<int> d1, vector<int> d2)
{
	vector<vector<int>> dp1(n, vector<int>(n, 0));
	vector<vector<int>> dp2(n, vector<int>(n, 0));
	int t1, t2;
	int temp1, temp2, temp3, temp4;
	for (int i = 0; i < n; i++) {
		if (d1[i] < d2[i]) {
			dp1[i][i] = d1[i];
			dp2[i][i] = 0;
		}
		else {
			dp2[i][i] = d2[i];
			dp1[i][i] = 0;
		}
	}
	for (int i = 1; i < n; i++) {
		int x = 0;
		int y = i;
		for (int j = 0; j < n - i; j++) {
			temp1 = dp1[x][y - 1];
			temp2 = dp2[x][y - 1];
			MachineHandling_judgement(d1[y], d2[y], temp1, temp2);
			temp3 = dp1[x + 1][y];
			temp4 = dp2[x + 1][y];
			MachineHandling_judgement(d1[x], d2[x], temp3, temp4);
			if (max(temp1, temp2) > max(temp3, temp4)) {
				dp1[x][y] = temp3;
				dp2[x][y] = temp4;
			}
			else if (max(temp1, temp2) < max(temp3, temp4)) {
				dp1[x][y] = temp1;
				dp2[x][y] = temp2;
			}
			else if (max(temp1, temp2) == max(temp3, temp4)) {
				if (max(temp1, temp2) > min(temp3, temp4)) {
					dp1[x][y] = temp3;
					dp2[x][y] = temp4;
				}
				else {
					dp1[x][y] = temp1;
					dp2[x][y] = temp2;
				}
			}
			++x;
			++y;
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d, %d  \t", dp1[i][j], dp2[i][j]);
		}
		cout << '\n';
	}
	return dp1[0][n - 1] > dp2[0][n - 1] ? dp1[0][n - 1] : dp2[0][n - 1];
}

void LClabuladong::MachineHandling_judgement(int t1, int t2, int& temp1, int& temp2)
{
	if (max(temp1, temp2 + t2) < max(temp1 + t1, temp2)) {
		temp2 = temp2 + t2;
	}
	else if (max(temp1, temp2 + t2) > max(temp1 + t1, temp2)) {
		temp1 = temp1 + t1;
	}
	else {
		if (min(temp1, temp2 + t2) < min(temp1 + t1, temp2)) {
			temp2 = temp2 + t2;
		}
		else {
			temp1 = temp1 + t1;
		}
	}
}

int LClabuladong::minRefuelStops(int target, int startFuel, vector<vector<int>>& stations)
{
	int cur_fuel = startFuel;
	int cur = 0;
	int ret = 0;
	priority_queue<int, vector<int>> fuel_stops;
	for (int i = 0; i < stations.size(); i++) {
		while (cur_fuel < stations[i][0] - cur) {
			if (fuel_stops.empty()) {
				if (cur + cur_fuel >= target) {
					return ret;
				}
				else {
					return -1;
				}
			}
			else {
				int add_fuel = fuel_stops.top();
				fuel_stops.pop();
				cur_fuel += add_fuel;
				++ret;
			}
		}
		cur_fuel -= (stations[i][0] - cur);
		cur = stations[i][0];
		fuel_stops.push(stations[i][1]);
		if (cur >= target) {
			return ret;
		}
	}
	while (cur_fuel < target - cur) {
		if (fuel_stops.empty()) {
			return -1;
		}
		else {
			int add_fuel = fuel_stops.top();
			fuel_stops.pop();
			cur_fuel += add_fuel;
			++ret;
		}
	}
	return ret;
}

int LClabuladong::robII(vector<int>& nums)
{
	int n = nums.size();
	if (n == 1) {
		return nums[0];
	}
	int ret = 0;
	vector<int> new_nums(2 * n, 0);
	n = new_nums.size();
	int old_n = n / 2;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < n / 2; j++) {
			new_nums[i * (n / 2) + j] = nums[j];
		}
	}
	vector<vector<int>> rob_dp(old_n - 1, vector<int>(n, 0));
	for (int i = 0; i < n; i++) {
		rob_dp[0][i] = new_nums[i];
	}
	for (int i = 1; i <= old_n - 2; i++) {
		int x = 0;
		int y = i;
		for (int j = 0; j < n - i; j++) {
			if (i - 2 < 0) {
				rob_dp[i][j] = max(max(rob_dp[i - 1][j], rob_dp[i - 1][j + 1]), max(new_nums[y], new_nums[x]));
			}
			else {
				rob_dp[i][j] = max(max(rob_dp[i - 1][j], rob_dp[i - 1][j + 1]), max(new_nums[y] + rob_dp[i - 2][j], new_nums[x] + rob_dp[i - 2][j + 2]));
			}
			++x;
			++y;
		}
	}
	for (int i = 0; i < old_n + 1; i++) {
		ret = max(ret, rob_dp[old_n - 2][i]);
	}

	return ret;
}

int LClabuladong::robIII(TreeNode* root)
{
	return robIII_cursion(root);
}

int LClabuladong::robIII_cursion(TreeNode* root)
{
	int left_max = 0;
	int right_max = 0;
	int left_left_max = 0;
	int left_right_max = 0;
	int right_left_max = 0;
	int right_right_max = 0;
	if (root->left) {
		left_max = robIII_cursion(root->left);
		if (root->left->left) {
			left_left_max = root->left->left->val;
		}
		if (root->left->right) {
			left_right_max = root->left->right->val;
		}
	}
	if (root->right) {
		right_max = robIII_cursion(root->right);
		if (root->right->left) {
			right_left_max = root->right->left->val;
		}
		if (root->right->right) {
			right_right_max = root->right->right->val;
		}
	}
	root->val = max(right_max + left_max, root->val + right_left_max + right_right_max + left_left_max + left_right_max);

	return root->val;
}

int LClabuladong::minCapability(vector<int>& nums, int k)
{
	int n = nums.size();
	int l = 1;
	int r = 1e9;
	/* 二分法求左侧边界 */
	while (l < r) {
		int m = (l + r) / 2;
		if (minCapability_check(nums, k, m)) {
			r = m;
		}
		else {
			l = m + 1;
		}
	}
	return l;
}

bool LClabuladong::minCapability_check(vector<int>& nums, int k, int m)
{
	int cnt = 0;
	int pre = -2;
	/* 不能连续选, 最大值为m, 看是否可以选够k个元素 */
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] <= m && i - pre > 1) {
			cnt++;
			pre = i;
		}
	}
	return cnt >= k;
}

int LClabuladong::maxProfit(vector<int>& prices)
{
	int n = prices.size();
	vector<int> pre_min(n);
	pre_min[0] = prices[0];
	int sell_price = prices[n - 1];
	int profit = 0;
	for (int i = 1; i < n; i++) {
		pre_min[i] = min(pre_min[i - 1], prices[i]);
	}
	for (int i = n - 2; i >= 0; i--) {
		profit = max(profit, sell_price - pre_min[i]);
		sell_price = max(sell_price, prices[i]);
	}
	return profit;
}

int LClabuladong::maxProfitII(vector<int>& prices)
{
	int buy_price = prices[0];
	int sell_price;
	int profit = 0;
	for (int i = 1; i < prices.size(); i++) {
		sell_price = prices[i];
		if (sell_price - buy_price > 0) {
			profit += (sell_price - buy_price);
		}
		buy_price = sell_price;
	}
	return profit;
}

int LClabuladong::maxProfitIII(vector<int>& prices)
{
	int n = prices.size();
	vector<vector<int>> dp_aux(n, vector<int>(n, 0));
	int ret = -1;

	for (int i = n - 2; i >= 0; i--) {
		for (int j = i + 1; j < n; j++) {
			dp_aux[i][j] = max(max(prices[j] - prices[i], 0), max(dp_aux[i + 1][j], dp_aux[i][j - 1]));
		}
	}
	for (int k = 0; k < n - 1; k++) {
		ret = max(ret, dp_aux[0][k] + dp_aux[k + 1][n - 1]);
	}
	return max(ret, dp_aux[0][n - 1]);
}

vector<int> LClabuladong::plusOne(vector<int>& digits)
{
	int n = digits.size();
	digits[n - 1] += 1;
	for (int i = n - 1; i >= 0; i--) {
		if (i == 0 && digits[i] == 10) {
			digits[i] -= 10;
			digits.push_back(0);
			for (int i = n; i >= 1; i--) {
				digits[i] = digits[i - 1];
			}
			digits[0] = 1;
		}
		if (digits[i] == 10) {
			digits[i] -= 10;
			digits[i - 1] += 1;
			continue;
		}
		else {
			break;
		}
	}
	return digits;
}

int LClabuladong::maxProfitIV(int k, vector<int>& prices)
{
	int n = prices.size();
	vector<vector<int>> dp(n, vector<int>(2 * k, 0));
	for (int i = 0; i < k; i++) {
		dp[0][2 * i] = -prices[0];
	}
	for (int i = 1; i < n; i++) {
		dp[i][0] = max(dp[i - 1][0], -prices[i]);
		for (int j = 1; j < 2 * k; j++) {
			if (j % 2 == 0) {
				dp[i][j] = max(dp[i - 1][j - 1] - prices[i], dp[i - 1][j]);
			}
			else {
				dp[i][j] = max(dp[i - 1][j - 1] + prices[i], dp[i - 1][j]);
			}
		}
	}
	return dp[n - 1][2 * k - 1];
}

int LClabuladong::maxProfitFreezing(vector<int>& prices)
{
	int ans = 0;
	int n = prices.size();
	int dp0 = -prices[0];//持股态
	int dp1 = 0;//不持股不冷冻态
	int dp2 = 0;//冷动态
	for (int i = 1; i < n; i++) {
		int a = max(dp1 - prices[i], dp0);//不持股不冷动—》持股，持股-》持股
		int b = max(dp1, dp2);//不持股不冷动-》不持股不冷动，冷冻-》不持股不冷动
		int c = dp0 + prices[i];//持股-》冷冻

		dp0 = a;
		dp1 = b;
		dp2 = c;

		ans = max(b, c);
	}

	return ans;
}

int LClabuladong::maxProfitV(vector<int>& prices, int fee)
{
	int holding = -prices[0];
	int unholding = 0;
	int a = 0, b = 0;
	for (int i = 0; i < prices.size(); i++) {
		a = unholding - prices[i];
		b = holding - fee + prices[i];

		holding = max(holding, a);
		unholding = max(unholding, b);
	}
	return max(holding, unholding);
}

int LClabuladong::strStr(string haystack, string needle)
{
	vector<int> kmp_arr(needle.size(), 0);
	for (int i = 2; i < needle.size(); i++) {
		kmp_arr[i] = strStr_getIndex(needle, kmp_arr, i);
	}
	int p_haystack = 0, p_needle = 0;
	while (p_needle < needle.size() && p_haystack < haystack.size()) {
		if (haystack[p_haystack] == needle[p_needle]) {
			++p_haystack;
			++p_needle;
		}
		else {
			if (p_needle != 0) {
				p_needle = kmp_arr[p_needle];
			}
			else {
				++p_haystack;
			}
		}
	}
	if (p_haystack == haystack.size() && p_needle < needle.size()) {
		return -1;
	}
	return p_haystack - needle.size();
}

int LClabuladong::strStr_getIndex(string& needle, vector<int>& kmp_arr, int cur)
{
	char temp = needle[cur - 1];
	int index = kmp_arr[cur - 1];
	while (true) {
		if (temp == needle[index]) {
			return index + 1;
		}
		else {
			if (index != 0) {
				index = kmp_arr[index];
			}
			else {
				break;
			}
		}
	}
	return 0;
}

int LClabuladong::eraseOverlapIntervals(vector<vector<int>>& intervals)
{
	sort(intervals.begin(), intervals.end());//用end来升序排更简单
	int ret = 0;
	int start = INT_MIN;
	int end = INT_MIN;
	int n = intervals.size();
	for (int i = 0; i < n; i++) {
		int t_start = intervals[i][0];
		int t_end = intervals[i][1];
		if (t_start >= end || t_end < end) {
			if (t_end < end) {
				++ret;
			}
			start = t_start;
			end = t_end;
		}
		else {
			++ret;
		}
	}
	return ret;
}

int LClabuladong::videoStitching(vector<vector<int>>& clips, int time)
{
	sort(clips.begin(), clips.end(), [](vector<int>& a, vector<int>& b)->bool
		{
			if (a[0] == b[0]) {
				return a[1] > b[1];
			}
			return a[0] < b[0];
		});
	int ret = 0;
	int cur_end = 0;
	int next_end = 0;
	int i = 0;
	int n = clips.size();
	while (i < n && clips[i][0] <= cur_end) {
		while (i < n && clips[i][0] <= cur_end) {
			next_end = max(next_end, clips[i][1]);
			i++;
		}
		++ret;
		cur_end = next_end;
		if (cur_end >= time) {
			return ret;
		}
	}
	return -1;
}

int LClabuladong::jump(vector<int>& nums)
{
	int ret = 1;
	int cur_jump;
	int next_jump = 0;
	int n = nums.size();
	if (n == 1) {
		return 0;
	}
	int i = 0;
	cur_jump = nums[i];
	while (i < n && cur_jump) {
		while (cur_jump) {
			++i;
			if (i >= n - 1) {
				return ret;
			}
			--cur_jump;
			--next_jump;
			next_jump = max(next_jump, nums[i]);
		}
		cur_jump = next_jump;
		++ret;
	}
	return ret;
}

bool LClabuladong::canJump(vector<int>& nums)
{
	int cur_jump = 0, next_jump = 0;
	int i = 0;
	int n = nums.size();
	if (n == 1) {
		return true;
	}
	cur_jump = nums[0];
	while (i < n && cur_jump) {
		while (cur_jump) {
			++i;
			--cur_jump;
			--next_jump;
			if (i < n)
				next_jump = max(next_jump, nums[i]);
			else
				return true;
		}
		cur_jump = next_jump;
	}

	if (i >= n - 1) {
		return true;
	}
	else {
		return false;
	}
}

int LClabuladong::totalNQueens(int n)
{
	totalNQueens_ret = 0;
	totalNQueens_map.resize(n);
	for (int i = 0; i < n; i++) {
		totalNQueens_map[i].resize(n, '.');
	}
	for (int i = 0; i < n; i++) {
		totalNQueens_map[0][i] = 'Q';
		totalNQueens_recursion(1);
		totalNQueens_map[0][i] = '.';
	}

	return totalNQueens_ret;
}

void LClabuladong::totalNQueens_recursion(int row)
{
	int n = totalNQueens_map.size();

	if (row == n) {
		totalNQueens_ret++;
		return;
	}

	for (int i = 0; i < n; i++) {
		totalNQueens_map[row][i] = 'Q';
		if (totalNQueens_mapIsOk(row, i)) {
			totalNQueens_recursion(row + 1);
		}
		totalNQueens_map[row][i] = '.';
	}
}

bool LClabuladong::totalNQueens_mapIsOk(int row, int col)
{
	int n = totalNQueens_map.size();
	for (int i = 0; i < row; i++) {
		if (totalNQueens_map[i][col] == 'Q') {
			return false;
		}
	}
	int i = row - 1, j = col - 1;
	while (i >= 0 && j >= 0) {
		if (totalNQueens_map[i][j] == 'Q') {
			return false;
		}
		--i;
		--j;
	}
	i = row - 1;
	j = col + 1;
	while (i >= 0 && j < n) {
		if (totalNQueens_map[i][j] == 'Q') {
			return false;
		}
		--i;
		++j;
	}
	return true;
}

vector<vector<int>> LClabuladong::combinationSum3(int k, int n)
{
	vector<vector<int>> ret;
	int v_sum = 0;
	vector<int> temp;
	combinationSum3_recursion(k, n, ret, v_sum, temp, 1);

	return ret;
}

void LClabuladong::combinationSum3_recursion(int k, int n, vector<vector<int>>& buckets, int& v_sum, vector<int>& cur, int x)
{
	if (cur.size() == k) {
		if (v_sum == n) {
			buckets.push_back(cur);
		}
		return;
	}

	for (int i = x; i < 10; i++) {
		if ((v_sum + i == n && cur.size() + 1 != k) || (cur.size() + 1 == k && v_sum + i != n)) {
			continue;
		}
		v_sum += i;
		cur.push_back(i);
		combinationSum3_recursion(k, n, buckets, v_sum, cur, i + 1);
		cur.pop_back();
		v_sum -= i;
	}
}

vector<vector<int>> LClabuladong::combinationSum2(vector<int>& candidates, int target)
{
	vector<int> cur;
	int sum = 0;
	sort(candidates.begin(), candidates.end(), less<int>());
	combinationSum2_backtrace(candidates, target, cur, sum, 0);

	return combinationSum2_ret;
}

void LClabuladong::combinationSum2_backtrace(vector<int>& candidates, int target, vector<int>& cur, int sum, int x)
{
	if (sum == target) {
		combinationSum2_ret.push_back(cur);
		return;
	}

	for (int i = x; i < candidates.size(); i++) {
		if (sum + candidates[i] > target) {
			continue;
		}
		sum += candidates[i];
		cur.push_back(candidates[i]);
		combinationSum2_backtrace(candidates, target, cur, sum, i + 1);
		cur.pop_back();
		sum -= candidates[i];
		while (i + 1 < candidates.size() && candidates[i + 1] == candidates[i]) {
			++i;
		}
	}
}

vector<vector<int>> LClabuladong::permuteUnique(vector<int>& nums)
{
	vector<int> cur;
	sort(nums.begin(), nums.end(), less<int>());
	vector<bool> nums_flags(nums.size(), true);

	permuteUnique_backtrace(nums, cur, nums_flags, 0);
	return permuteUnique_ret;
}

void LClabuladong::permuteUnique_backtrace(vector<int>& nums, vector<int>& cur, vector<bool>& nums_flags, int x)
{
	if (x == nums.size()) {
		permuteUnique_ret.push_back(cur);
		return;
	}
	int last_value = 11;

	for (int i = 0; i < nums.size(); i++) {
		if (!nums_flags[i]) {
			continue;
		}
		if (nums[i] == last_value) {
			continue;
		}
		cur.push_back(nums[i]);
		nums_flags[i] = false;
		permuteUnique_backtrace(nums, cur, nums_flags, x + 1);
		nums_flags[i] = true;
		cur.pop_back();
		last_value = nums[i];
	}
}

vector<vector<int>> LClabuladong::subsetsWithDup(vector<int>& nums)
{
	sort(nums.begin(), nums.end(), less<int>());
	vector<int> cur;
	subsetsWithDup_backtrace(nums, cur, 0);

	return subsetsWithDup_ret;
}

void LClabuladong::subsetsWithDup_backtrace(vector<int>& nums, vector<int>& cur, int x)
{
	subsetsWithDup_ret.push_back(cur);
	if (x == nums.size()) {
		return;
	}
	int last_select = nums[x] + 1;
	for (int i = x; i < nums.size(); i++) {
		if (last_select == nums[i]) {
			continue;
		}
		cur.push_back(nums[i]);
		subsetsWithDup_backtrace(nums, cur, i + 1);
		cur.pop_back();
		last_select = nums[i];
	}
}

void LClabuladong::solveSudoku(vector<vector<char>>& board)
{
	solveSudoku_IsOk = false;
	vector<vector<bool>> grid_flag(9, vector<bool>(10, true));
	solveSuduku_InitGrid(board, grid_flag);
	int grid_start, grid_index;
	solveSuduku_BackTrace(board, grid_flag, 0, 0);
	board = solveSudoku_ret;
}

void LClabuladong::solveSuduku_InitGrid(vector<vector<char>>& board, vector<vector<bool>>& grid_flag)
{
	int grid_start, grid_index;
	for (int i = 0; i < 9; i++) {
		grid_start = (i - i % 3) * 9 + i % 3 * 3;
		for (int j = 0; j < 9; j++) {
			grid_index = grid_start + (j - (j % 3)) * 3 + j % 3;
			int grid_x = (grid_index - (grid_index % 9)) / 9;
			int grid_y = grid_index % 9;
			if (board[grid_x][grid_y] != '.') {
				grid_flag[i][board[grid_x][grid_y] - '0'] = false;
			}
		}
	}
}

void LClabuladong::solveSuduku_BackTrace(vector<vector<char>>& board, vector<vector<bool>>& grid_flag, int grid_num, int number)
{
	if (grid_num == 9) {
		solveSudoku_ret = board;
		solveSudoku_IsOk = true;
		return;
	}

	int grid_start, grid_index, k, j;
	grid_start = (grid_num - grid_num % 3) * 9 + grid_num % 3 * 3;
	grid_index = grid_start + (number - (number % 3)) * 3 + number % 3;
	int grid_x = (grid_index - (grid_index % 9)) / 9;
	int grid_y = grid_index % 9;

	if (board[grid_x][grid_y] != '.') {
		if (number == 8) {
			solveSuduku_BackTrace(board, grid_flag, grid_num + 1, 0);
		}
		else {
			solveSuduku_BackTrace(board, grid_flag, grid_num, number + 1);
		}
		return;
	}

	for (k = 1; k <= 9; k++) {
		if (!grid_flag[grid_num][k] || !solveSuduku_RemedyIsOk(board, grid_x, grid_y, k)) {
			continue;
		}
		board[grid_x][grid_y] = '0' + k;
		grid_flag[grid_num][k] = false;
		if (number == 8) {
			solveSuduku_BackTrace(board, grid_flag, grid_num + 1, 0);
		}
		else {
			solveSuduku_BackTrace(board, grid_flag, grid_num, number + 1);
		}
		if (solveSudoku_IsOk) {
			return;
		}
		grid_flag[grid_num][k] = true;
		board[grid_x][grid_y] = '.';
	}
}

bool LClabuladong::solveSuduku_RemedyIsOk(vector<vector<char>>& board, int x, int y, int value)
{
	for (int i = 0; i < 9; i++) {
		if (board[x][i] - '0' == value) {
			return false;
		}
	}
	for (int i = 0; i < 9; i++) {
		if (board[i][y] - '0' == value) {
			return false;
		}
	}
	return true;
}

vector<string> LClabuladong::generateParenthesis(int n)
{
	int left_num = 0;
	int right_num = 0;
	string cur(n * 2, '.');
	generateParenthesis_BackTrace(n, 0, cur, left_num, right_num);
	return generateParenthesis_ret;
}

void LClabuladong::generateParenthesis_BackTrace(int n, int index, string& cur, int left_num, int right_num)
{
	if (right_num == n) {
		generateParenthesis_ret.push_back(cur);
		return;
	}

	if (right_num == left_num) {
		cur[index] = '(';
		++left_num;
		generateParenthesis_BackTrace(n, index + 1, cur, left_num, right_num);
		--left_num;
		cur[index] = '.';
		return;
	}
	if (left_num < n) {
		cur[index] = '(';
		++left_num;
		generateParenthesis_BackTrace(n, index + 1, cur, left_num, right_num);
		--left_num;
		cur[index] = '.';
	}

	if (right_num < n) {
		cur[index] = ')';
		++right_num;
		generateParenthesis_BackTrace(n, index + 1, cur, left_num, right_num);
		--right_num;
		cur[index] = '.';
	}

	return;
}

LClabuladong::LClabuladong(ListNode* head)
{
	getRandom_head = head;
	srand(time(0));
}

//蓄水池随机
int LClabuladong::getRandom()
{
	ListNode* p = getRandom_head;
	int ret;
	int cnt = 0;
	while (p) {
		cnt++;
		if (rand() % cnt == 0) {
			ret = p->val;
		}
		p = p->next;
	}
	return ret;
}

LClabuladong::LClabuladong(vector<int>& nums)
{
	shuffle_arr = nums;
	srand(time(NULL));
}

vector<int> LClabuladong::reset()
{
	return shuffle_arr;
}

//洗牌算法
vector<int> LClabuladong::shuffle()
{
	vector<int> rt = shuffle_arr;
	int n = rt.size();

	//通过将数组从n-1向前随机（包括自身）交换val，实现数组的打乱
	for (int i = n - 1; i >= 0; --i) {
		int idx = rand() % (i + 1);
		int tmp = rt[idx];
		rt[idx] = rt[i];
		rt[i] = tmp;
	}
	return rt;
}

int LClabuladong::RandomPick(int target)
{
	int index;
	for (int i = 0, cur = 0; i < shuffle_arr.size(); i++) {
		if (shuffle_arr[i] == target) {
			cur++;
			if (rand() % cur == 0) {
				index = cur;
			}
		}
	}
	return index;
}

int LClabuladong::singleNumber(vector<int>& nums)
{
	int ret = 0;
	for (int& i : nums) {
		ret ^= i;
	}
	return ret;
}

int LClabuladong::hammingWeight(uint32_t n)
{
	int res = 0;
	while (n != 0) {
		n = n & (n - 1);
		res++;
	}
	return res;
}

bool LClabuladong::isPowerOfTwo(int n)
{
	if (n <= 0) {
		return false;
	}
	return (n & (n - 1)) == 0;
}

int LClabuladong::missingNumber(vector<int>& nums)
{
	int n = nums.size();
	int sum = n * (n + 1) / 2;
	for (int& i : nums) {
		sum -= i;
	}
	return sum;
}

long long LClabuladong::trailingZeroes(long long n)
{
	long long res = 0;
	for (long long d = n; d / 5 > 0; d = d / 5) {
		res += d / 5;
	}
	return res;
}

int LClabuladong::preimageSizeFZF(int k)
{
	return preimageSizeFZF_rightBound(k) - preimageSizeFZF_LeftBound(k) + 1;
}

long long LClabuladong::preimageSizeFZF_LeftBound(int k)
{
	long long lo = 0, hi = LLONG_MAX;
	long long mid;

	while (hi > lo) {
		mid = (hi + lo) / 2;
		long long numZero = trailingZeroes(mid);
		if (numZero >= k) {
			hi = mid;
		}
		else if (numZero < k) {
			lo = mid + 1;
		}
	}
	return hi;
}

long long LClabuladong::preimageSizeFZF_rightBound(int k)
{
	long long lo = 0, hi = LLONG_MAX;
	long long mid;

	while (hi > lo) {
		mid = hi - (hi - lo) / 2;
		long long numZero = trailingZeroes(mid);
		if (numZero > k) {
			hi = mid - 1;
		}
		else if (numZero <= k) {
			lo = mid;
		}
	}
	return hi;
}

int LClabuladong::countPrimes(int n)
{
	if (n == 0) {
		return 0;
	}
	bool* isPrime = new bool[n];
	memset(isPrime, 1, sizeof(bool) * n);

	for (int i = 2; i < n; i++) {
		if (isPrime[i]) {
			for (int j = i * 2; j < n; j += i) {
				isPrime[j] = false;
			}
		}
	}

	int ret = 0;
	for (int i = 2; i < n; i++) {
		if (isPrime[i])
			++ret;
	}

	return ret;
}

int LClabuladong::superPow(int a, vector<int>& b)
{
	int ans = 1;
	for (int e : b) {
		ans = (long)pow1(ans, 10) * pow1(a, e) % superPow_mod;
	}
	return ans;
}

int LClabuladong::pow1(int x, int n)
{
	int res = 1;
	while (n) {
		if (n % 2) {
			res = (long)res * x % superPow_mod;
		}
		x = (long)x * x % superPow_mod;
		n /= 2;
	}
	return res;
}

vector<int> LClabuladong::findErrorNums(vector<int>& nums)
{
	vector<int> ret(2, 0);
	vector<int> countArr(nums.size() + 1, 0);
	for (int i = 0; i < nums.size(); i++) {
		countArr[nums[i]]++;
	}
	for (int i = 1; i < countArr.size(); i++) {
		if (countArr[i] == 0) {
			ret[1] = i;
		}
		if (countArr[i] == 2) {
			ret[0] = i;
		}
	}
	return ret;
}

bool LClabuladong::canWinNim(int n)
{
	return n % 4 != 0;
}

int LClabuladong::bulbSwitch(int n)
{
	return (int)sqrt(n);
}

vector<int> LClabuladong::diffWaysToCompute(string expression)
{
	vector<string> exp = diffWaysToCompute_GetOpVector(expression);
	if (exp.size() <= 3) {
		diffWaysToCompute_ret = diffWaysToCompute_GetComupute(exp, 0, exp.size() - 1);
	}
	else {
		diffWaysToCompute_GetComupute(exp, 0, exp.size() - 1);
	}

	return diffWaysToCompute_ret;
}

vector<int> LClabuladong::diffWaysToCompute_GetComupute(vector<string> expression, int begin, int end)
{
	vector<int> factor1, factor2, ans;
	if (begin == end) {
		ans.push_back(atoi(expression[begin].c_str()));
		return ans;
	}
	char op;
	if (end - begin == 2) {
		op = expression[begin + 1][0];
		switch (op)
		{
		case '+':
			ans.push_back(atoi(expression[begin].c_str()) + atoi(expression[end].c_str()));
			break;
		case '-':
			ans.push_back(atoi(expression[begin].c_str()) - atoi(expression[end].c_str()));
			break;
		case '*':
			ans.push_back(atoi(expression[begin].c_str()) * atoi(expression[end].c_str()));
			break;
		}
		return ans;
	}

	int mid;

	for (mid = begin; mid <= end - 2; mid += 2) {
		factor1 = diffWaysToCompute_GetComupute(expression, begin, mid);
		factor2 = diffWaysToCompute_GetComupute(expression, mid + 2, end);
		op = expression[mid + 1][0];
		int t;
		switch (op)
		{
		case '+':
			for (int i = 0; i < factor1.size(); i++) {
				for (int j = 0; j < factor2.size(); j++) {
					t = factor1[i] + factor2[j];
					ans.push_back(t);
				}
			}
			break;
		case '-':
			for (int i = 0; i < factor1.size(); i++) {
				for (int j = 0; j < factor2.size(); j++) {
					t = factor1[i] - factor2[j];
					ans.push_back(t);
				}
			}
			break;
		case '*':
			for (int i = 0; i < factor1.size(); i++) {
				for (int j = 0; j < factor2.size(); j++) {
					t = factor1[i] * factor2[j];
					ans.push_back(t);
				}
			}
			break;
		}
	}
	if (begin == 0 && end == expression.size() - 1) {
		diffWaysToCompute_ret = ans;
	}
	return ans;
}

vector<string> LClabuladong::diffWaysToCompute_GetOpVector(string expression)
{
	int start = 0;
	string op;
	string t_str;
	vector<string> ret;
	for (int i = 0; i < expression.size(); i++) {
		if (expression[i] == '+' || expression[i] == '-' || expression[i] == '*') {
			t_str = expression.substr(start, i - start);
			ret.push_back(t_str);
			ret.push_back(expression.substr(i, 1));
			start = i + 1;
		}
	}
	t_str = expression.substr(start, expression.size() - start);
	ret.push_back(t_str);
	return ret;
}

int LClabuladong::removeCoveredIntervals(vector<vector<int>>& intervals)
{
	auto cmp = [](vector<int> a, vector<int> b) -> bool {
		if (a[0] != b[0]) {
			return a[0] < b[0];
		}
		else {
			return a[1] > b[1];
		}
		};
	sort(intervals.begin(), intervals.end(), cmp);
	int ret = intervals.size();

	int start = intervals[0][0], end = intervals[0][1];
	int t_start, t_end;
	for (int i = 1; i < intervals.size(); i++) {
		t_end = intervals[i][1];
		if (t_end <= end) {
			--ret;
			continue;
		}
		end = t_end;
	}

	return ret;
}

vector<vector<int>> LClabuladong::merge(vector<vector<int>>& intervals)
{
	vector<vector<int>> ret;
	auto cmp = [](vector<int>& a, vector<int>& b)->bool {
		if (a[0] != b[0]) {
			return a[0] < b[0];
		}
		else {
			return a[1] < b[1];
		}
		};
	sort(intervals.begin(), intervals.end(), cmp);
	int start = intervals[0][0], end = intervals[0][1];
	int t_start, t_end;
	for (int i = 1; i < intervals.size(); i++) {
		t_start = intervals[i][0];
		t_end = intervals[i][1];
		if (t_start <= end) {
			end = max(end, t_end);
		}
		else {
			vector<int> t_ve = { start, end };
			ret.push_back(t_ve);
			start = t_start;
			end = t_end;
		}
	}
	vector<int> t_ve = { start, end };
	ret.push_back(t_ve);
	return ret;
}

vector<vector<int>> LClabuladong::intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList)
{
	vector<vector<int>> ret;
	if (firstList.size() == 0 || secondList.size() == 0) {
		return ret;
	}
	int curStart = firstList[0][0], curEnd = firstList[0][1];
	int firstIndex = 1, secondIndex = 0;
	bool compareFirst = false;
	int tStart, tEnd;
	int left, right;

	while (firstIndex <= firstList.size() && secondIndex <= secondList.size()) {
		if (compareFirst) {
			if (firstIndex == firstList.size()) {
				break;
			}
			tStart = firstList[firstIndex][0];
			tEnd = firstList[firstIndex][1];
			firstIndex++;
		}
		else {
			if (secondIndex == secondList.size()) {
				break;
			}
			tStart = secondList[secondIndex][0];
			tEnd = secondList[secondIndex][1];
			secondIndex++;
		}
		left = max(tStart, curStart);
		right = min(tEnd, curEnd);
		if (left > right) {
			if (tStart > curEnd) {
				curStart = tStart;
				curEnd = tEnd;
				compareFirst = !compareFirst;
			}
			continue;
		}
		vector<int> tVector = { left, right };
		ret.push_back(tVector);
		if (tEnd > curEnd) {
			curStart = tStart;
			curEnd = tEnd;
			compareFirst = !compareFirst;
		}
		else if (tEnd < curEnd) {
			continue;
		}
		else {
			if (tStart > curStart) {
				curStart = tStart;
				curEnd = tEnd;
				compareFirst = !compareFirst;
			}
		}
	}

	return ret;
}

bool LClabuladong::isPossible(vector<int>& nums)
{
	map<int, int> countMap;
	for (int i = 0; i < nums.size(); i++) {
		if (!countMap.count(nums[i])) {
			countMap[nums[i]] = 1;
			continue;
		}
		countMap[nums[i]]++;
	}

	int startIndex = countMap.begin()->first;
	int lastNum = 0;
	int continuousQuanlity = 0;
	int lastValue;


	while (countMap[startIndex] != 0) {
		lastNum = 0;
		continuousQuanlity = 0;
		lastValue = startIndex - 1;
		bool startIndexIsUpdata = false;
		for (auto it = countMap.find(startIndex); it != countMap.end(); it++) {
			if (it->first != lastValue + 1) {
				startIndex = it->first;
				break;
			}
			if (it->second >= lastNum) {
				lastNum = it->second;
				continuousQuanlity++;
				it->second--;
				lastValue = it->first;
				if (it->second != 0 && !startIndexIsUpdata) {
					startIndex = it->first;
					startIndexIsUpdata = true;
				}
			}
			else {
				break;
			}

		}
		if (continuousQuanlity < 3) {
			return false;
		}
	}

	return true;
}

vector<int> LClabuladong::pancakeSort(vector<int>& arr)
{
	vector<int> ret;
	for (int n = arr.size(); n > 1; n--) {
		int index = max_element(arr.begin(), arr.begin() + n) - arr.begin();
		if (index == n - 1) {
			continue;
		}
		reverse(arr.begin(), arr.begin() + index + 1);
		reverse(arr.begin(), arr.begin() + n);
		ret.push_back(index + 1);
		ret.push_back(n);
	}
	return ret;
}

int LClabuladong::calculate(string s)
{
	stack<int> ops;
	ops.push(1);
	int sign = 1;

	int ret = 0;
	int n = s.length();
	int i = 0;
	while (i < n) {
		if (s[i] == ' ') {
			i++;
		}
		else if (s[i] == '+') {
			sign = ops.top();
			i++;
		}
		else if (s[i] == '-') {
			sign = -ops.top();
			i++;
		}
		else if (s[i] == '(') {
			ops.push(sign);
			i++;
		}
		else if (s[i] == ')') {
			ops.pop();
			i++;
		}
		else {
			long num = 0;
			while (i < n && s[i] >= '0' && s[i] <= '9') {
				num = num * 10 + s[i] - '0';
				i++;
			}
			ret += sign * num;
		}
	}
	return ret;
}

int LClabuladong::trap(vector<int>& height)
{
	int left = 0, right;
	int n = height.size();
	int startValue, endValue = 0;
	int endIndex = -1;
	int sum = 0;
	int curSum = 0;
	while (left < n) {
		if (height[left] == 0) {
			++left;
			continue;
		}
		startValue = height[left];
		break;
	}
	right = left + 1;
	while (right < n) {
		if (height[right] >= startValue) {
			sum += curSum;
			cout << "在" << startValue << "与" << height[right] << "之前，加上" << curSum << endl;
			startValue = height[right];
			left = right;
			right = left + 1;
			curSum = 0;
			endValue = 0;
			continue;
		}
		if (height[right] >= endValue) {
			endValue = height[right];
			endIndex = right;
		}
		int difference = startValue - height[right];
		curSum += difference;
		++right;
		if (right == n) {
			if (endIndex != n - 1) {
				curSum = trap_recalculation(height, left, endIndex);
				cout << "在" << height[left] << "与" << height[endIndex] << "之前，加上" << curSum << endl;
				sum += curSum;
				left = endIndex;
				right = left + 1;
				endValue = 0;
				continue;
			}
			else {
				curSum = trap_recalculation(height, left, endIndex);
				sum += curSum;
				break;
			}
		}
	}

	return sum;
}

int LClabuladong::trap_recalculation(vector<int> height, int startIndex, int endIndex)
{
	int sum = 0;
	for (int i = startIndex + 1; i < endIndex; i++) {
		sum += (height[endIndex] - height[i]);
	}

	return sum;
}

bool LClabuladong::isRectangleCover(vector<vector<int>>& rectangles)
{
	int x1, y1, x2, y2;
	int X1 = INT_MAX, Y1 = INT_MAX, X2 = -1, Y2 = -1;
	int expectArea = 0;
	map<pair<int, int>, int> points;
	pair<int, int> p1, p2, p3, p4;
	for (auto& r : rectangles) {
		x1 = r[0], y1 = r[1], x2 = r[2], y2 = r[3];
		expectArea += (x2 - x1) * (y2 - y1);
		X1 = min(X1, x1);
		Y1 = min(Y1, y1);
		X2 = max(X2, x2);
		Y2 = max(Y2, y2);
		p1 = { x1, y1 };
		p2 = { x1, y2 };
		p3 = { x2, y1 };
		p4 = { x2, y2 };
		for (auto& p : { p1, p2, p3, p4 }) {
			if (!points.count(p)) {
				points[p] = 1;
			}
			else {
				points.erase(p);
			}
		}
	}
	int curArea = (X2 - X1) * (Y2 - Y1);
	if (curArea != expectArea) {
		return false;
	}
	if (points.size() != 4) {
		return false;
	}
	p1 = { X1, Y1 };
	p2 = { X1, Y2 };
	p3 = { X2, Y1 };
	p4 = { X2, Y2 };
	for (auto& p : { p1, p2, p3, p4 }) {
		if (!points.count(p)) {
			return false;
		}
	}
	return true;
}

LClabuladong::ExamRoom::ExamRoom(int n)
{
	nextIndex = 0;
	seats.resize(n, false);
	diffOfSeats.resize(n, INT_MAX);
	num = 0;
}

int LClabuladong::ExamRoom::seat()
{
	int ret = nextIndex;
	num++;
	int curDiff = -1;
	seats[nextIndex] = true;
	diffOfSeats[nextIndex] = 0;
	int index = nextIndex;
	while (1) {
		index++;
		if (index >= seats.size()) {
			break;
		}
		if (diffOfSeats[index] <= index - nextIndex) {
			break;
		}
		diffOfSeats[index] = index - nextIndex;
	}
	index = nextIndex;
	while (1) {
		index--;
		if (index < 0) {
			break;
		}
		if (diffOfSeats[index] <= nextIndex - index) {
			break;
		}
		diffOfSeats[index] = nextIndex - index;
	}

	for (int i = 0; i < diffOfSeats.size(); i++) {
		if (diffOfSeats[i] > curDiff) {
			curDiff = diffOfSeats[i];
			nextIndex = i;
		}
		if (diffOfSeats[i] == curDiff) {
			nextIndex = min(nextIndex, i);
		}
	}

	return ret;
}

void LClabuladong::ExamRoom::leave(int p)
{
	seats[p] = false;
	int index = p;
	int left, right;
	num--;
	if (num == 0) {
		nextIndex = 0;
		diffOfSeats = vector<int>(diffOfSeats.size(), INT_MAX);
		return;
	}
	while (1) {
		index++;
		if (index >= seats.size()) {
			right = seats.size();
			break;
		}
		if (diffOfSeats[index] < diffOfSeats[index - 1]) {
			right = index - 1;
			break;
		}
		if (diffOfSeats[index] == diffOfSeats[index - 1]) {
			right = index;
			break;
		}
	}
	index = p;
	while (1) {
		index--;
		if (index < 0) {
			left = -1;
			break;
		}
		if (diffOfSeats[index] < diffOfSeats[index + 1]) {
			left = index + 1;
			break;
		}
		if (diffOfSeats[index] == diffOfSeats[index + 1]) {
			left = index;
			break;
		}
	}
	if (left == -1) {
		for (int i = right - 1; i >= 0; i--) {
			diffOfSeats[i] = diffOfSeats[i + 1] + 1;
		}
		if (diffOfSeats[0] >= diffOfSeats[nextIndex]) {
			nextIndex = 0;
		}
		return;
	}
	if (right == diffOfSeats.size()) {
		for (int i = left + 1; i < right; i++) {
			diffOfSeats[i] = diffOfSeats[i - 1] + 1;
		}
		if (diffOfSeats[right - 1] > diffOfSeats[nextIndex]) {
			nextIndex = right - 1;
		}
		return;
	}

	for (int i = left + 1; i < right; i++) {
		diffOfSeats[i] = diffOfSeats[i - 1] + 1;
	}

	for (int i = right - 1; i > left; i--) {
		if (diffOfSeats[i] == diffOfSeats[i + 1] || diffOfSeats[i] == diffOfSeats[i + 1] + 1) {
			if (diffOfSeats[i] > diffOfSeats[nextIndex]) {
				nextIndex = i;
			}
			if (diffOfSeats[i] == diffOfSeats[nextIndex]) {
				nextIndex = min(i, nextIndex);
			}
			break;
		}
		diffOfSeats[i] = diffOfSeats[i + 1] + 1;
	}
}

bool LClabuladong::isSubsequence(string s, string t)
{
	int tIndex = 0, sIndex = 0;
	while (tIndex != t.size() && sIndex != s.size()) {
		if (s[sIndex] == t[tIndex]) {
			sIndex++;
		}
		tIndex++;
	}

	if (sIndex == s.size()) {
		return true;
	}
	return false;
}

int LClabuladong::numMatchingSubseq(string s, vector<string>& words)
{
	int ret = 0;
	vector<vector<int>> sRecord(26);
	for (int i = 0; i < s.size(); i++) {
		sRecord[s[i] - 'a'].push_back(i);
	}

	for (string& word : words) {
		int count = -1;
		for (int i = 0; i < word.size(); i++) {
			vector<int>& charRecord = sRecord[word[i] - 'a'];
			count = numMatchingSubseq_FindNext(charRecord, count);
			if (count == -1) {
				break;
			}
		}
		if (count != -1) {
			ret++;
		}
	}

	return ret;
}

int LClabuladong::numMatchingSubseq_FindNext(vector<int>& charRecord, int count)
{
	if (charRecord.empty()) {
		return -1;
	}
	if (count == -1) {
		return charRecord[0];
	}
	int left = 0, right = charRecord.size();
	while (left < right) {
		int mid = (right - left) / 2 + left;
		if (charRecord[mid] > count) {
			right = mid;
		}
		else {
			left = mid + 1;
		}
	}

	if (left == charRecord.size()) {
		return -1;
	}
	else {
		return charRecord[left];
	}
}

vector<int> LClabuladong::sortedSquares(vector<int>& nums)
{
	int left, right;
	int start = 0;
	int minNum = INT_MAX;
	for (int i = 0; i < nums.size(); i++) {
		nums[i] = nums[i] * nums[i];
		if (nums[i] < minNum) {
			minNum = nums[i];
			start = i;
		}
	}

	vector<int> ret(nums.size(), 0);
	left = start - 1;
	right = start + 1;
	ret[0] = nums[start];
	int index = 1;

	while (index < nums.size()) {
		if (left < 0) {
			ret[index] = nums[right];
			index++;
			right++;
			continue;
		}
		if (right >= nums.size()) {
			ret[index] = nums[left];
			index++;
			left--;
			continue;
		}
		if (nums[left] < nums[right]) {
			ret[index] = nums[left];
			left--;
		}
		else {
			ret[index] = nums[right];
			right++;
		}
		index++;
	}

	return ret;
}

bool LClabuladong::isPerfectSquare(int num)
{
	int left = 1, right = num;

	while (left <= right) {
		int mid = (right - left) / 2 + left;
		int x = mid * mid;
		if (x < num) {
			left = mid + 1;
		}
		else if (x > num) {
			right = mid - 1;
		}
		else {
			return true;
		}
	}

	return false;
}

bool LClabuladong::backspaceCompare(string s, string t)
{
	int sIndex = s.size() - 1, tIndex = t.size() - 1;
	while (sIndex >= 0 && tIndex >= 0) {
		if (s[sIndex] == '#') {
			int numBack = 0;
			while (sIndex >= 0 && s[sIndex] == '#') {
				numBack++;
				sIndex--;
			}
			while (sIndex >= 0 && numBack) {
				if (s[sIndex] != '#') {
					numBack--;
				}
				else {
					numBack++;
				}
				sIndex--;
			}

			continue;
		}
		if (t[tIndex] == '#') {
			int numBack = 0;
			while (tIndex >= 0 && t[tIndex] == '#') {
				numBack++;
				tIndex--;
			}
			while (tIndex >= 0 && numBack) {
				if (t[tIndex] != '#') {
					numBack--;
				}
				else {
					numBack++;
				}
				tIndex--;
			}

			continue;
		}
		if (s[sIndex] == t[tIndex]) {
			--sIndex;
			--tIndex;
		}
		else {
			return false;
		}
	}

	while (tIndex >= 0) {
		if (t[tIndex] == '#') {
			int numBack = 0;
			while (tIndex >= 0 && t[tIndex] == '#') {
				numBack++;
				tIndex--;
			}
			while (tIndex >= 0 && numBack) {
				if (t[tIndex] != '#') {
					numBack--;
				}
				else {
					numBack++;
				}
				tIndex--;
			}
		}
		else {
			return false;
		}
	}
	while (sIndex >= 0) {
		if (s[sIndex] == '#') {
			int numBack = 0;
			while (sIndex >= 0 && s[sIndex] == '#') {
				numBack++;
				sIndex--;
			}
			while (sIndex >= 0 && numBack) {
				if (s[sIndex] != '#') {
					numBack--;
				}
				else {
					numBack++;
				}
				sIndex--;
			}
		}
		else {
			return false;
		}
	}
	if (sIndex >= 0 || tIndex >= 0) {
		return false;
	}
	return true;
}

int LClabuladong::totalFruit(vector<int>& fruits)
{
	int left = 0, right = 0;
	int ret = 0;
	int maxFruits = 0;
	int lf = -1, rf = -1;
	int numType = 0;
	while (right < fruits.size()) {
		maxFruits = right - left;
		rf = -1;
		while (right < fruits.size()) {
			int thisType = fruits[right];


			if (thisType == lf || thisType == rf) {
				maxFruits++;
				right++;
			}
			else {
				if (numType < 2) {
					if (numType == 0) {
						lf = thisType;
					}
					else {
						rf = thisType;
					}
					numType++;
					maxFruits++;
					right++;
				}
				else {
					break;
				}
			}

			if (thisType != fruits[left]) {
				left = right - 1;
			}
		}
		ret = max(maxFruits, ret);
		lf = fruits[left];
		numType = 1;
	}
	return ret;
}

vector<vector<int>> LClabuladong::generateMatrix(int n)
{
	vector<vector<int>> ret(n, vector<int>(n, -1));
	generateMatrix_Padding(ret, 0, 0, 1, 3);
	return ret;
}

void LClabuladong::generateMatrix_Padding(vector<vector<int>>& matrix, int x, int y, int value, int n)
{
	if (n <= 0) {
		return;
	}

	matrix[x][y] = value++;
	for (int i = 0; i < n - 1; i++) {
		y++;
		matrix[x][y] = value++;
	}
	for (int i = 0; i < n - 1; i++) {
		x++;
		matrix[x][y] = value++;
	}
	for (int i = 0; i < n - 1; i++) {
		y--;
		matrix[x][y] = value++;
	}
	for (int i = 0; i < n - 2; i++) {
		x--;
		matrix[x][y] = value++;
	}

	generateMatrix_Padding(matrix, x, y + 1, value, n - 2);
}

vector<int> LClabuladong::spiralOrder(vector<vector<int>>& matrix)
{
	int width = matrix[0].size(), height = matrix.size();
	vector<int> ret(width * height, -1);
	int index = 0;

	int xStart = 0, yStart = 0;

	while (width > 0 && height > 0) {

		for (int i = 0; i < width - 1; i++) {
			ret[index++] = matrix[xStart][yStart];
			yStart++;
		}

		if (index == ret.size() - 1) {
			ret[index] = matrix[xStart][yStart];
			return ret;
		}

		for (int i = 0; i < height - 1; i++) {
			ret[index++] = matrix[xStart][yStart];
			xStart++;
		}

		if (index == ret.size() - 1) {
			ret[index] = matrix[xStart][yStart];
			return ret;
		}

		for (int i = 0; i < width - 1; i++) {
			ret[index++] = matrix[xStart][yStart];
			yStart--;
		}

		for (int i = 0; i < height - 1; i++) {
			ret[index++] = matrix[xStart][yStart];
			xStart--;
		}

		height -= 2;
		width -= 2;
		xStart += 1;
		yStart += 1;
	}

	return ret;
}

ListNode* LClabuladong::removeElements(ListNode* head, int val)
{
	if (!head) {
		return head;
	}
	ListNode* newHead = new ListNode();
	newHead->next = head;
	ListNode* p = newHead;
	ListNode* temp;
	while (p->next) {
		if (p->next->val == val) {
			temp = p->next;
			p->next = temp->next;
			delete temp;
		}
		else {
			p = p->next;
		}
	}
	head = newHead->next;

	return head;
}

LClabuladong::MyLinkedList::MyLinkedList()
{
	head = new Node(0);
	size = 0;
}

int LClabuladong::MyLinkedList::get(int index)
{
	if (index >= size) {
		return -1;
	}
	Node* p = head->next;
	while (index) {
		p = p->next;
		index--;
	}
	return p->val;
}

void LClabuladong::MyLinkedList::addAtHead(int val)
{
	Node* temp = new Node(val);
	temp->next = head->next;
	temp->prev = head;
	if (head->next) {
		head->next->prev = temp;
	}
	head->next = temp;
	size++;
}

void LClabuladong::MyLinkedList::addAtTail(int val)
{
	Node* p = head;
	while (p->next) {
		p = p->next;
	}
	Node* temp = new Node(val);
	temp->prev = p;
	temp->next = NULL;
	p->next = temp;
	size++;
}

void LClabuladong::MyLinkedList::addAtIndex(int index, int val)
{
	if (index > size) {
		return;
	}
	Node* p = head;
	while (index) {
		p = p->next;
		index--;
	}
	Node* temp = new Node(val);
	temp->next = p->next;
	temp->prev = p;
	if (temp->next) {
		temp->next->prev = temp;
	}
	p->next = temp;
	size++;
}

void LClabuladong::MyLinkedList::deleteAtIndex(int index)
{
	if (index >= size) {
		return;
	}
	Node* p = head->next;
	while (index) {
		p = p->next;
		index--;
	}
	p->prev->next = p->next;
	if (p->next) {
		p->next->prev = p->prev;
	}
	delete p;
	size--;
}

bool LClabuladong::isAnagram(string s, string t)
{
	unordered_map<char, int> sMap, tMap;
	for (char& c : s) {
		if (sMap.count(c)) {
			sMap[c]++;
		}
		else {
			sMap[c] = 1;
		}
	}
	for (char& c : t) {
		if (tMap.count(c)) {
			tMap[c]++;
		}
		else {
			tMap[c] = 1;
		}
	}

	for (pair<char, int> p : tMap) {
		if (!sMap.count(p.first)) {
			return false;
		}
		sMap[p.first] -= p.second;
	}
	for (pair<char, int> p : sMap) {
		if (p.second != 0) {
			return false;
		}
	}

	return true;
}

bool LClabuladong::canConstruct(string ransomNote, string magazine)
{
	int a[26] = { 0 };
	for (char& c : magazine) {
		a[c - 'a']++;
	}
	for (char& c : ransomNote) {
		if (a[c - 'a'] == 0) {
			return false;
		}
		a[c - 'a']--;
	}
	return true;
}

vector<int> LClabuladong::intersection(vector<int>& nums1, vector<int>& nums2)
{
	unordered_map<int, int> retMap;
	for (int& i : nums1) {
		if (retMap.count(i)) {
			continue;
		}
		retMap[i] = 1;
	}
	for (int& i : nums2) {
		if (retMap.count(i)) {
			retMap[i] = 2;
		}
	}
	vector<int> ret;
	for (auto& i : retMap) {
		if (i.second == 2) {
			ret.push_back(i.first);
		}
	}
	return ret;
}

vector<int> LClabuladong::intersect(vector<int>& nums1, vector<int>& nums2)
{
	unordered_map<int, int> retMap;
	vector<int> ret;
	for (int& i : nums1) {
		if (retMap.count(i)) {
			retMap[i] += 1;
			continue;
		}
		retMap[i] = 1;
	}
	for (int& i : nums2) {
		if (retMap.count(i)) {
			ret.push_back(i);
			if (--retMap[i] == 0) {
				retMap.erase(i);
			}
		}
	}
	return ret;
}

bool LClabuladong::isHappy(int n)
{
	string num;
	unordered_set<int> thisSet;
	int total = n;
	while (1) {
		if (thisSet.count(total)) {
			return false;
		}
		num = to_string(total);
		total = 0;
		for (char& i : num) {
			total += pow(i - '0', 2);
		}
		if (total == 1) {
			return true;
		}
		thisSet.insert(atoi(num.c_str()));
	}
}

vector<int> LClabuladong::twoSum(vector<int>& nums, int target)
{
	unordered_map<int, int> thisMap;
	for (int i = 0; i < nums.size(); i++) {
		if (thisMap.count(target - nums[i])) {
			return vector<int>({ thisMap[target - nums[i]], i });
		}
		thisMap[nums[i]] = i;
	}
	return vector<int>();
}

int LClabuladong::fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4)
{
	int n = nums1.size();
	unordered_map<int, int> thisMap;
	for (int& i : nums1) {
		if (thisMap.count(i)) {
			thisMap[i]++;
			continue;
		}
		thisMap[i] = 1;
	}
	unordered_map<int, int> thisMap2;
	for (int& i : nums2) {
		for (auto p : thisMap) {
			if (thisMap2.count(i + p.first)) {
				thisMap2[i + p.first] += p.second;
				continue;
			}
			thisMap2[i + p.first] = p.second;
		}
	}
	thisMap = thisMap2;
	thisMap2.clear();
	for (int& i : nums3) {
		for (auto p : thisMap) {
			if (thisMap2.count(i + p.first)) {
				thisMap2[i + p.first] += p.second;
				continue;
			}
			thisMap2[i + p.first] = p.second;
		}
	}
	thisMap = thisMap2;
	int ret = 0;
	for (int& i : nums4) {
		if (thisMap.count(-i)) {
			ret += thisMap[-i];
		}
	}
	return ret;
}

vector<vector<int>> LClabuladong::threeSum(vector<int>& nums)
{
	vector<vector<int>> result;
	sort(nums.begin(), nums.end());
	// 找出a + b + c = 0
	// a = nums[i], b = nums[j], c = -(a + b)
	for (int i = 0; i < nums.size(); i++) {
		// 排序之后如果第一个元素已经大于零，那么不可能凑成三元组
		if (nums[i] > 0) {
			break;
		}
		if (i > 0 && nums[i] == nums[i - 1]) { //三元组元素a去重
			continue;
		}
		unordered_set<int> set;
		for (int j = i + 1; j < nums.size(); j++) {
			if (j > i + 2
				&& nums[j] == nums[j - 1]
				&& nums[j - 1] == nums[j - 2]) { // 三元组元素b去重
				continue;
			}
			int c = 0 - (nums[i] + nums[j]);
			if (set.find(c) != set.end()) {
				result.push_back({ nums[i], nums[j], c });
				set.erase(c);// 三元组元素c去重
			}
			else {
				set.insert(nums[j]);
			}
		}
	}
	return result;
}

vector<vector<int>> LClabuladong::fourSum(vector<int>& nums, int target)
{
	vector<vector<int>> ans;
	sort(nums.begin(), nums.end(), less<int>());
	int left, right;
	for (int i = 0; i < nums.size() - 3; i++) {
		if (i != 0 && nums[i] == nums[i - 1]) {
			continue;
		}
		for (int j = i + 1; j < nums.size() - 2; j++) {
			if (j != i + 1 && nums[j] == nums[j - 1]) {
				continue;
			}
			left = j + 1;
			right = nums.size() - 1;
			while (left < right) {
				int t = nums[i] + nums[j] + nums[left] + nums[right];
				if (t < target) {
					left++;
				}
				else if (t > target) {
					right--;
				}
				else {
					ans.push_back(vector<int>({ nums[i], nums[j], nums[left], nums[right] }));
					while (left < right && nums[left + 1] == nums[left]) left++;
					while (left < right && nums[right - 1] == nums[right]) right--;
					left++;
					right--;
				}
			}
		}
	}
	return ans;
}

void LClabuladong::reverseString(vector<char>& s)
{
	int left, right;
	left = 0;
	right = s.size() - 1;
	char t;
	while (left < right) {
		t = s[left];
		s[left] = s[right];
		s[right] = t;
		left++;
		right--;
	}
}

string LClabuladong::reverseStr(string s, int k)
{
	string ret(s.size(), '0');
	int n = s.size();
	int bound_begin = -1;
	int bound_index;
	int ans_index = 0;
	while (bound_begin + 1 < n) {
		bound_index = bound_begin + k;
		if (bound_index >= n) {
			bound_index = n - 1;
		}
		while (bound_index > bound_begin) {
			ret[ans_index++] = s[bound_index--];
		}
		bound_begin += (2 * k);
		if (bound_begin >= n) {
			bound_begin = n - 1;
		}
		for (int i = bound_index + k + 1; i <= bound_begin; i++) {
			ret[ans_index++] = s[i];
		}
	}

	return ret;
}

string LClabuladong::reverseWords(string s)
{
	string ans;
	vector<string> words = reverseWords_GetWords(s);
	for (int i = words.size() - 1; i >= 0; i--) {
		ans += words[i];
		if (i != 0) {
			ans += ' ';
		}
	}

	return ans;
}

vector<string> LClabuladong::reverseWords_GetWords(string s)
{
	vector<string> ret;
	string tWord;
	for (char& c : s) {
		if (c == ' ') {
			if (tWord.size() == 0) {
				continue;
			}
			else {
				ret.push_back(tWord);
				tWord.clear();
			}
		}
		else {
			tWord += c;
		}
	}
	if (tWord.size() != 0) {
		ret.push_back(tWord);
	}

	return ret;
}

bool LClabuladong::repeatedSubstringPattern(string s)
{
	string str = s + s;
	str = str.substr(1, str.size() - 2);
	if (str.find(s) == -1)
		return false;
	return true;
}

string LClabuladong::removeDuplicates(string s)
{
	string ret;
	stack<int> strStack;
	for (char& c : s) {
		if (strStack.size() == 0 || strStack.top() != c) {
			strStack.push(c);
		}
		else {
			strStack.pop();
		}
	}
	ret.resize(strStack.size());
	for (int i = ret.size() - 1; i >= 0; i--) {
		ret[i] = strStack.top();
		strStack.pop();
	}

	return ret;
}

int LClabuladong::evalRPN(vector<string>& tokens)
{
	stack<int> valStack;
	for (int i = 0; i < tokens.size(); i++) {
		if ((tokens[i][0] >= '0' && tokens[i][0] <= '9') || (tokens[i][0] == '-' && tokens[i].size() > 1)) {
			valStack.push(atoi(tokens[i].c_str()));
		}
		else {
			int b = valStack.top();
			valStack.pop();
			int a = valStack.top();
			valStack.pop();
			valStack.push(evalRPN_Compute(a, b, tokens[i][0]));
		}
	}
	return valStack.top();
}

int LClabuladong::evalRPN_Compute(int a, int b, char op)
{
	int ret = 0;
	switch (op)
	{
	case '+':
		return a + b;
	case '-':
		return a - b;
	case '*':
		return a * b;
	case '/':
		return a / b;
	default:
		break;
	}

	return 0;
}

vector<int> LClabuladong::topKFrequent(vector<int>& nums, int k)
{
	vector<int> ret;
	unordered_map<int, int> numFreq;
	for (int i = 0; i < nums.size(); i++) {
		if (!numFreq.count(nums[i])) {
			numFreq[nums[i]] = 1;
		}
		else {
			numFreq[nums[i]] += 1;
		}
	}
	//定义比较函数
	struct cmp
	{
		bool operator()(const pair<int, int> a, const pair<int, int> b)
		{
			return a.second < b.second;
		}
	};
	priority_queue<pair<int, int>, vector<pair<int, int>>, cmp>  pq;
	for (auto p : numFreq) {
		pq.push(p);
	}
	ret.resize(k);
	for (int i = 0; i < k; i++) {
		ret[i] = pq.top().first;
		pq.pop();
	}

	return ret;
}

vector<int> LClabuladong::postorderTraversal(TreeNode* root)
{
	if (root) {
		postorderTraversal_recursion(root);
	}

	return postorderTraversal_ret;
}

void LClabuladong::postorderTraversal_recursion(TreeNode* root)
{
	if (root->left) {
		postorderTraversal_recursion(root->left);
	}
	if (root->right) {
		postorderTraversal_recursion(root->right);
	}
	postorderTraversal_ret.push_back(root->val);
}

vector<vector<int>> LClabuladong::levelOrder(TreeNode* root)
{
	vector<vector<int>> ret;
	queue<TreeNode*> nodeQueue;
	unordered_map<TreeNode*, int> nodeLevel;
	if (!root) {
		return ret;
	}
	nodeQueue.push(root);
	nodeLevel[root] = 1;
	while (!nodeQueue.empty()) {
		TreeNode* tNode = nodeQueue.front();
		int tLevel = nodeLevel[tNode];
		nodeQueue.pop();
		if (ret.size() < tLevel) {
			ret.push_back(vector<int>());
		}
		ret[tLevel - 1].push_back(tNode->val);
		if (tNode->left) {
			nodeQueue.push(tNode->left);
			nodeLevel[tNode->left] = tLevel + 1;
		}
		if (tNode->right) {
			nodeQueue.push(tNode->right);
			nodeLevel[tNode->right] = tLevel + 1;
		}
	}

	return ret;
}

vector<int> LClabuladong::preorder(Node* root)
{
	vector<int> ret;
	stack<Node*> nodeStack;
	if (!root) {
		return ret;
	}
	nodeStack.push(root);
	while (!nodeStack.empty()) {
		Node* tNode = nodeStack.top();
		nodeStack.pop();
		ret.push_back(tNode->val);
		for (int i = tNode->children.size() - 1; i >= 0; i--) {
			nodeStack.push(tNode->children[i]);
		}
	}
	return ret;
}

vector<int> LClabuladong::postorder(Node* root)
{
	vector<int> ret;
	stack<Node*> nodeStack;
	if (!root) {
		return ret;
	}
	nodeStack.push(root);
	while (!nodeStack.empty()) {
		Node* tNode = nodeStack.top();
		nodeStack.pop();
		ret.push_back(tNode->val);
		for (int i = 0; i < tNode->children.size(); i++) {
			nodeStack.push(tNode->children[i]);
		}
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

bool LClabuladong::isSymmetric(TreeNode* root)
{
	if (root == NULL) return true;
	return isSymmetric_Compare(root->left, root->right);
}

bool LClabuladong::isSymmetric_Compare(TreeNode* left, TreeNode* right) {
	if (left == NULL && right != NULL) return false;
	else if (left != NULL && right == NULL) return false;
	else if (left == NULL && right == NULL) return true;
	else if (left->val != right->val) return false;
	else return isSymmetric_Compare(left->left, right->right) && isSymmetric_Compare(left->right, right->left);

}

bool LClabuladong::isSameTree(TreeNode* p, TreeNode* q)
{
	if (p == NULL && q == NULL) return true;
	if (p == NULL && q != NULL) return false;
	if (p != NULL && q == NULL) return false;
	if (p->val != q->val) return false;
	return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

bool LClabuladong::isSubtree(TreeNode* root, TreeNode* subRoot)
{
	return isSubtree_Contain(root, subRoot);
}

bool LClabuladong::isSubtree_Contain(TreeNode* root, TreeNode* subRoot)
{
	if (isSameTree(root, subRoot)) {
		return true;
	}
	if (root == NULL) {
		return false;
	}
	return isSubtree_Contain(root->left, subRoot) || isSubtree_Contain(root->right, subRoot);
}

bool LClabuladong::isBalanced(TreeNode* root)
{
	if (root == NULL) {
		return true;
	}
	else {
		return abs(isBalanced_GetDiff(root->left) - isBalanced_GetDiff(root->right)) <= 1 && isBalanced(root->left) && isBalanced(root->right);
	}
}

int LClabuladong::isBalanced_GetDiff(TreeNode* root)
{
	if (root == NULL) {
		return 0;
	}
	else {
		return max(isBalanced_GetDiff(root->left), isBalanced_GetDiff(root->right)) + 1;
	}
}

vector<string> LClabuladong::binaryTreePaths(TreeNode* root)
{
	return binaryTreePaths_GetPath(root);
}

vector<string> LClabuladong::binaryTreePaths_GetPath(TreeNode* root)
{
	vector<string> ret;
	vector<string> left, right;
	string val = to_string(root->val);
	if (root->left) {
		left = binaryTreePaths_GetPath(root->left);
	}
	if (root->right) {
		right = binaryTreePaths_GetPath(root->right);
	}
	ret.insert(ret.end(), left.begin(), left.end());
	ret.insert(ret.end(), right.begin(), right.end());
	if (ret.size() == 0) {
		ret.push_back(val);
		return ret;
	}
	for (string& s : ret) {
		s = val + "->" + s;
	}

	return ret;
}

int LClabuladong::sumOfLeftLeaves(TreeNode* root)
{
	sumOfLeftLeaves_ret = 0;
	sumOfLeftLeaves_Recursion(root);
	return sumOfLeftLeaves_ret;
}

void LClabuladong::sumOfLeftLeaves_Recursion(TreeNode* root)
{
	if (root->left && root->left->left == NULL && root->left->right == NULL) {
		sumOfLeftLeaves_ret += root->left->val;
	}
	if (root->left) {
		sumOfLeftLeaves_Recursion(root->left);
	}
	if (root->right) {
		sumOfLeftLeaves_Recursion(root->right);
	}
}

int LClabuladong::findBottomLeftValue(TreeNode* root)
{
	findBottomLeftValue_h = -1;
	findBottomLeftValue_Recursion(root, 1);
	return findBottomLeftValue_ret;
}

void LClabuladong::findBottomLeftValue_Recursion(TreeNode* root, int h)
{
	if (h > findBottomLeftValue_h) {
		findBottomLeftValue_h = h;
		findBottomLeftValue_ret = root->val;
	}
	if (root->left) {
		findBottomLeftValue_Recursion(root->left, h + 1);
	}
	if (root->right) {
		findBottomLeftValue_Recursion(root->right, h + 1);
	}
}

bool LClabuladong::hasPathSum(TreeNode* root, int targetSum)
{
	if (!root) {
		return false;
	}
	return hasPathSum_Recursion(root, targetSum);
}

bool LClabuladong::hasPathSum_Recursion(TreeNode* root, int targetSum)
{
	if (root->left == NULL && root->right == NULL && root->val == targetSum) {
		return true;
	}
	bool left = false, right = false;
	if (root->left) {
		left = hasPathSum_Recursion(root->left, targetSum - root->val);
	}
	if (root->right) {
		right = hasPathSum_Recursion(root->right, targetSum - root->val);
	}
	return left || right;
}

int LClabuladong::getMinimumDifference(TreeNode* root)
{
	return getMinimumDifference_Recursion(root);
}

int LClabuladong::getMinimumDifference_Recursion(TreeNode* root)
{
	int left, right;
	if (root->left) {
		left = root->val - getMinimumDifference_GetMax(root->left);
		left = min(left, getMinimumDifference_Recursion(root->left));
	}
	else
		left = INT_MAX;
	if (root->right) {
		right = getMinimumDifference_GetMin(root->right) - root->val;
		right = min(right, getMinimumDifference_Recursion(root->right));
	}
	else
		right = INT_MAX;
	return min(left, right);
}

int LClabuladong::getMinimumDifference_GetMax(TreeNode* root)
{
	if (root->right == NULL) {
		return root->val;
	}
	else {
		return getMinimumDifference_GetMax(root->right);
	}
}

int LClabuladong::getMinimumDifference_GetMin(TreeNode* root)
{
	if (root->left == NULL) {
		return root->val;
	}
	else {
		return getMinimumDifference_GetMin(root->left);
	}
}

vector<int> LClabuladong::findMode(TreeNode* root)
{
	vector<int> ret;
	vector<int> nodeList;
	findMode_TreeToVector(root, nodeList);
	int num = 0;
	int thisLen = -1;
	int lastVal = INT_MAX;
	for (int i = 0; i < nodeList.size(); i++) {
		if (nodeList[i] == lastVal) {
			thisLen++;
		}
		else {
			if (thisLen > num) {
				ret.clear();
				num = thisLen;
				ret.push_back(lastVal);
			}
			else if (thisLen == num) {
				ret.push_back(lastVal);
			}
			thisLen = 1;
			lastVal = nodeList[i];
		}
	}
	if (thisLen > num) {
		ret.clear();
		num = thisLen;
		ret.push_back(lastVal);
	}
	else if (thisLen == num) {
		ret.push_back(lastVal);
	}

	return ret;
}

void LClabuladong::findMode_TreeToVector(TreeNode* root, vector<int>& nodeList)
{
	if (root->left)
		findMode_TreeToVector(root->left, nodeList);
	nodeList.push_back(root->val);
	if (root->right)
		findMode_TreeToVector(root->right, nodeList);
}

TreeNode* LClabuladong::sortedArrayToBST(vector<int>& nums)
{
	return sortedArrayToBST_Traverse(nums);
}

TreeNode* LClabuladong::sortedArrayToBST_Traverse(vector<int>& nums)
{
	int n = nums.size();
	if (n == 1) {
		return new TreeNode(nums[0]);
	}
	int mid = n / 2;
	TreeNode* ret = new TreeNode(nums[mid]);
	vector<int> left(nums.begin(), nums.begin() + mid);
	vector<int> right(nums.begin() + mid + 1, nums.end());
	if (left.size() > 0)
		ret->left = sortedArrayToBST_Traverse(left);
	if (right.size() > 0)
		ret->right = sortedArrayToBST_Traverse(right);
	return ret;
}

vector<string> LClabuladong::restoreIpAddresses(string s)
{
	string ip;
	restoreIpAddresses_Traverse(s, ip, 0);
	return restoreIpAddresses_ret;
}

void LClabuladong::restoreIpAddresses_Traverse(string s, string cur, int n)
{
	if (n == 3) {
		if (s.size() == 0 || (s[0] == '0' && s.size() != 1)) {
			return;
		}
		if (atoi(s.c_str()) >= 0 && atoi(s.c_str()) <= 255) {
			cur += s;
			restoreIpAddresses_ret.push_back(cur);
		}
		return;
	}
	for (int i = 1; i <= 3; i++) {
		if (s.size() < i) {
			return;
		}
		string num = s.substr(0, i);
		if (num[0] == '0' && num.size() != 1) {
			return;
		}
		if (atoi(num.c_str()) >= 0 && atoi(num.c_str()) <= 255) {
			string temp = cur + num + '.';
			restoreIpAddresses_Traverse(s.substr(i, s.size() - i), temp, n + 1);
		}
	}
}

vector<vector<int>> LClabuladong::findSubsequences(vector<int>& nums)
{
	vector<int> cur;
	findSubsequences_Traverse(nums, 0, cur);
	return findSubsequences_ret;
}

void LClabuladong::findSubsequences_Traverse(vector<int>& nums, int begin, vector<int> cur)
{
	if (begin == nums.size()) {
		return;
	}
	int used[201] = { 0 };
	for (int i = begin; i < nums.size(); i++) {
		if (used[nums[i] + 100] == 1) {
			continue;
		}
		if (cur.size() == 0 || nums[i] >= cur[cur.size() - 1]) {
			cur.push_back(nums[i]);
			if (cur.size() > 1) {
				findSubsequences_ret.push_back(cur);
			}
			findSubsequences_Traverse(nums, i + 1, cur);
			used[nums[i] + 100] = 1;
			cur.pop_back();
		}
	}
}

vector<string> LClabuladong::findItinerary(vector<vector<string>>& tickets)
{
	findItinerary_targets.clear();
	vector<string> result;
	for (const vector<string>& vec : tickets) {
		findItinerary_targets[vec[0]][vec[1]]++; // 记录映射关系
	}
	result.push_back("JFK"); // 起始机场
	findItinerary_Traverse(tickets.size(), result);
	return result;
}

bool LClabuladong::findItinerary_Traverse(int ticketNum, vector<string>& result)
{
	if (result.size() == ticketNum + 1) {
		return true;
	}
	for (pair<const string, int>& target : findItinerary_targets[result[result.size() - 1]]) {
		if (target.second > 0) { // 记录到达机场是否飞过了
			result.push_back(target.first);
			target.second--;
			if (findItinerary_Traverse(ticketNum, result)) return true;
			result.pop_back();
			target.second++;
		}
	}
	return false;
}

int LClabuladong::findContentChildren(vector<int>& g, vector<int>& s)
{
	int ret = 0;
	sort(g.begin(), g.end(), less<int>());
	sort(s.begin(), s.end(), less<int>());
	int gIndex = 0, sIndex = 0;
	while (gIndex < g.size() && sIndex < s.size()) {
		if (s[sIndex] >= g[gIndex]) {
			ret++;
			sIndex++;
			gIndex++;
		}
		else {
			sIndex++;
		}
	}

	return ret;
}

int LClabuladong::wiggleMaxLength(vector<int>& nums)
{
	if (nums.size() == 1) {
		return 1;
	}
	int begin = 1;
	while (begin < nums.size() && nums[begin] == nums[begin - 1]) {
		begin++;
	}
	if (begin >= nums.size()) {
		return 1;
	}
	int ret = 1;
	bool findBiger = true;
	if (nums[begin] < nums[begin - 1]) {
		findBiger = false;
	}
	for (int i = begin; i < nums.size(); i++) {
		if (findBiger) {
			while (nums[i] >= nums[i - 1]) {
				i++;
				if (i >= nums.size()) {
					break;
				}
			}
			ret++;
			findBiger = !findBiger;
			i--;
		}
		else {
			while (nums[i] <= nums[i - 1]) {
				i++;
				if (i >= nums.size()) {
					break;
				}
			}
			ret++;
			findBiger = !findBiger;
			i--;
		}
	}

	return ret;
}

int LClabuladong::largestSumAfterKNegations(vector<int>& nums, int k)
{
	int ret = 0;
	sort(nums.begin(), nums.end(), less<int>());
	int i = 0;
	for (i = 0; i < nums.size(); i++) {
		if (k == 0) {
			break;
		}
		if (nums[i] < 0) {
			nums[i] = -nums[i];
			k--;
		}
		else {
			break;
		}
	}
	if (i != 0 && i != nums.size()) {
		i = nums[i] > nums[i - 1] ? i - 1 : i;
	}
	else if (i == nums.size()) {
		i = i - 1;
	}
	for (int j = 0; j < nums.size(); j++) {
		ret += nums[j];
	}
	if (k % 2 != 0) {
		ret -= 2 * nums[i];
	}

	return ret;
}

int LClabuladong::canCompleteCircuit(vector<int>& gas, vector<int>& cost)
{
	int ret = -1;
	int sum = -1;
	int allSum = 0;
	for (int i = 0; i < gas.size(); i++) {
		allSum += (gas[i] - cost[i]);
		if (sum < 0) {
			ret = i;
			sum = gas[i] - cost[i];
			continue;
		}
		sum += (gas[i] - cost[i]);
	}
	if (allSum < 0) {
		return -1;
	}
	return ret;
}

int LClabuladong::candy(vector<int>& ratings)
{
	int n = ratings.size();
	int ret = 0;
	vector<int> right_left(n, 1);
	vector<int> left_right(n, 1);
	for (int i = 1; i < n; i++) {
		if (ratings[i] > ratings[i - 1]) {
			left_right[i] = left_right[i - 1] + 1;
		}
	}
	for (int i = n - 2; i >= 0; i--) {
		if (ratings[i] > ratings[i + 1]) {
			right_left[i] = right_left[i + 1] + 1;
		}
	}
	for (int i = 0; i < n; i++) {
		ret += max(right_left[i], left_right[i]);
	}

	return ret;
}

bool LClabuladong::lemonadeChange(vector<int>& bills)
{
	unordered_map<int, int> wallet;
	wallet[5] = 0;
	wallet[10] = 0;
	for (int i = 0; i < bills.size(); i++) {
		if (bills[i] == 5) {
			wallet[5] += 1;
		}
		else if (bills[i] == 10) {
			wallet[5] -= 1;
			wallet[10] += 1;
			if (wallet[5] < 0) {
				return false;
			}
		}
		else if (bills[i] == 20) {
			if ((wallet[10] == 0 && wallet[5] < 3) || (wallet[10] > 0 && wallet[5] == 0)) {
				return false;
			}
			if (wallet[10] > 0) {
				wallet[10] -= 1;
				wallet[5] -= 1;
			}
			else {
				wallet[5] -= 3;
			}
		}
	}
	return true;
}

vector<vector<int>> LClabuladong::reconstructQueue(vector<vector<int>>& people)
{
	struct cmp
	{
		bool operator()(vector<int>& a, vector<int>& b) {
			if (a[0] == b[0]) {
				return a[1] < b[1];
			}
			return a[0] > b[0];
		}
	};
	sort(people.begin(), people.end(), cmp());
	vector<vector<int>> ret;
	for (int i = 0; i < people.size(); i++) {
		ret.insert(ret.begin() + people[i][1], people[i]);//这用链表一定是更快的
	}

	return ret;
}

int LClabuladong::findMinArrowShots(vector<vector<int>>& points)
{
	struct cmp {
		bool operator()(vector<int>& a, vector<int>& b) {
			if (a[0] == b[0]) {
				return a[1] < b[1];
			}
			return a[0] < b[0];
		}
	};
	sort(points.begin(), points.end(), cmp());
	long left = LONG_MAX, right = LONG_MIN;
	int ret = 0;
	for (int i = 0; i < points.size(); i++) {
		if (points[i][0] > right) {
			ret++;
			left = points[i][0], right = points[i][1];
			continue;
		}
		left = points[i][0];
		right = min(right, points[i][1]);
	}

	return ret;
}

vector<int> LClabuladong::partitionLabels(string s)
{
	int n = s.size();
	vector<vector<int>> labels(26, vector<int>(2, -1));

	for (int i = 0; i < n; i++) {
		if (labels[s[i] - 'a'][0] == -1) {
			labels[s[i] - 'a'][0] = i;
		}
	}
	for (int i = n - 1; i >= 0; i--) {
		if (labels[s[i] - 'a'][1] == -1) {
			labels[s[i] - 'a'][1] = i;
		}
	}
	struct cmp {
		bool operator()(vector<int>& a, vector<int>& b) {
			if (a[0] == b[0]) {
				return a[1] < b[1];
			}
			return a[0] < b[0];
		}
	};
	sort(labels.begin(), labels.end(), cmp());
	int begin, end;
	begin = 0, end = 0;
	vector<int> ret;
	for (int i = 0; i < 26; i++) {
		if (labels[i][0] == -1) {
			continue;
		}
		if (labels[i][0] > end) {
			ret.push_back(end - begin + 1);
			begin = labels[i][0];
			end = labels[i][1];
		}
		else {
			if (labels[i][1] > end) {
				end = labels[i][1];
			}
		}
	}
	ret.push_back(end - begin + 1);
	return ret;
}

int LClabuladong::monotoneIncreasingDigits(int n)
{
	string strN = to_string(n);
	int i = 1;
	while (i < strN.length() && strN[i - 1] <= strN[i]) {
		i += 1;
	}
	if (i < strN.length()) {
		while (i > 0 && strN[i - 1] > strN[i]) {
			strN[i - 1] -= 1;
			i -= 1;
		}
		for (i += 1; i < strN.length(); ++i) {
			strN[i] = '9';
		}
	}
	return stoi(strN);
}

int LClabuladong::minCameraCover(TreeNode* root)
{
	minCameraCover_ret = 0;
	if (minCameraCover_recursion(root) == 0) { // root 无覆盖
		minCameraCover_ret++;
	}
	return minCameraCover_ret;
}

int LClabuladong::minCameraCover_recursion(TreeNode* cur)
{
	if (cur == NULL) return 2;
	int left = minCameraCover_recursion(cur->left);    // 左
	int right = minCameraCover_recursion(cur->right);  // 右
	if (left == 2 && right == 2) return 0;
	else if (left == 0 || right == 0) {
		minCameraCover_ret++;
		return 1;
	}
	else return 2;
}

int LClabuladong::minCostClimbingStairs(vector<int>& cost)
{
	int n = cost.size() + 1;
	vector<int> dp(n, 0);
	for (int j = 2; j < n; j++) {
		dp[j] = min(dp[j - 1] + cost[j - 1], dp[j - 2] + cost[j - 2]);
	}
	return dp[n - 1];
}

int LClabuladong::uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
{
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();
	if (obstacleGrid[m - 1][n - 1] == 1 || obstacleGrid[0][0] == 1) {
		return 0;
	}
	vector<vector<int>> dp = obstacleGrid;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (dp[i][j] == 1) {
				dp[i][j] = -1;
			}
		}
	}
	dp[0][0] = 1;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (dp[i][j] == -1) {
				continue;
			}
			if (i == 0 && j == 0) {
				continue;
			}
			dp[i][j] = ((j - 1 >= 0 && dp[i][j - 1] != -1) ? dp[i][j - 1] : 0) + ((i - 1 >= 0 && dp[i - 1][j] != -1) ? dp[i - 1][j] : 0);
		}
	}
	return dp[m - 1][n - 1];
}

int LClabuladong::integerBreak(int n)
{
	auto AutoMax = [](vector<int>& dp, int k)->int {
		int ret = -1;
		for (int i = 1; i < k; i++) {
			ret = max(ret, i * (k - i));
			ret = max(ret, dp[i] * (k - i));
		}
		return ret;
		};
	vector<int>dp(n + 1, 0);
	dp[2] = 1;
	for (int i = 3; i < n + 1; i++) {
		dp[i] = AutoMax(dp, i);
	}
	return dp[n];
}

int LClabuladong::lastStoneWeightII(vector<int>& stones)
{
	int n = stones.size();
	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += stones[i];
	}
	int m = sum / 2;
	vector<vector<int>> dp(n + 1, vector<int>(m + 1));
	dp[0][0] = true;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j <= m; ++j) {
			if (j < stones[i]) {
				dp[i + 1][j] = dp[i][j];
			}
			else {
				dp[i + 1][j] = dp[i][j] || dp[i][j - stones[i]];
			}
		}
	}
	for (int j = m;; --j) {
		if (dp[n][j]) {
			return sum - 2 * j;
		}
	}
}

int LClabuladong::findMaxForm(vector<string>& strs, int m, int n)
{
	int len = strs.size();
	vector<pair<int, int>> newStrs(len, pair<int, int>(0, 0));
	for (int i = 0; i < len; i++) {
		newStrs[i] = findMaxForm_CountZeroOne(strs[i]);
	}

	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

	for (int j = 0; j <= m; j++) {
		for (int k = 0; k <= n; k++) {
			if (j - newStrs[0].first < 0 || k - newStrs[0].second < 0) {
				continue;
			}
			dp[j][k] = 1;
		}
	}


	for (int i = 1; i < len; i++) {
		pair<int, int> curStr = newStrs[i];
		for (int j = m; j >= 0; j--) {
			for (int k = n; k >= 0; k--) {
				if (j - curStr.first < 0 || k - curStr.second < 0) {
					continue;
				}
				dp[j][k] = max(dp[j - curStr.first][k - curStr.second] + 1, dp[j][k]);
			}
		}

	}

	return dp[m][n];
}

pair<int, int> LClabuladong::findMaxForm_CountZeroOne(string s)
{
	int zeroNum = 0, oneNum = 0;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '0') {
			zeroNum++;
		}
		else if (s[i] == '1') {
			oneNum++;
		}
	}

	return pair<int, int>(zeroNum, oneNum);
}

int LClabuladong::findLengthOfLCIS(vector<int>& nums)
{
	int ret = 0;
	int curLen = 0, curMax = INT_MAX;
	curMax = nums[0];
	curLen = 1;
	for (int i = 1; i < nums.size(); i++) {
		if (nums[i] > curMax) {
			curLen++;
		}
		else {
			ret = max(ret, curLen);
			curLen = 1;
		}
		curMax = nums[i];
	}
	ret = max(ret, curLen);
	return ret;
}

int LClabuladong::findLength(vector<int>& nums1, vector<int>& nums2)
{
	int ret = 0;
	int n = nums1.size(), m = nums2.size();
	vector<vector<int>> arr(n, vector<int>(m, 0));
	for (int j = 0; j < m; j++) {
		arr[0][j] = nums1[0] == nums2[j];
		ret = max(ret, arr[0][j]);
	}
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (nums1[i] == nums2[j]) {
				if (j == 0) {
					arr[i][j] = 1;
				}
				else {
					arr[i][j] = arr[i - 1][j - 1] + 1;
				}
			}
			else {
				arr[i][j] = 0;
			}
			ret = max(ret, arr[i][j]);
		}
	}

	return ret;
}

int LClabuladong::maxUncrossedLines(vector<int>& nums1, vector<int>& nums2)
{
	int n = nums1.size(), m = nums2.size();
	vector<vector<int>> arr(n, vector<int>(m, 0));
	vector<int> toolArr(m, 0);
	vector<int> ttoolArr(m, 0);
	for (int i = 0; i < m; i++) {
		if (nums2[i] == nums1[0]) {
			arr[0][i] = 1;
			toolArr[i] = max(toolArr[i], arr[0][i]);
			for (int k = i + 1; k < m; k++) {
				toolArr[k] = max(toolArr[k], toolArr[i]);
			}
		}
	}
	for (int k = 0; k < m; k++) {
		ttoolArr[k] = toolArr[k];
	}
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (nums1[i] == nums2[j]) {
				if (j - 1 >= 0) {
					arr[i][j] = toolArr[j - 1] + 1;
				}
				else {
					arr[i][j] = 1;
				}
				ttoolArr[j] = max(toolArr[j], arr[i][j]);
				for (int k = j + 1; k < m; k++) {
					ttoolArr[k] = max(ttoolArr[k], ttoolArr[j]);
				}
			}
		}
		for (int k = 0; k < m; k++) {
			toolArr[k] = ttoolArr[k];
		}
	}
	int ret = 0;
	for (int& i : toolArr) {
		ret = max(i, ret);
	}

	return ret;
}

int LClabuladong::numDistinct(string s, string t)
{
	int n = s.size(), m = t.size();
	vector<vector<long>> dp(n, vector<long>(m, 0));
	if (s[0] == t[0]) {
		dp[0][0] = 1;
	}
	for (int i = 1; i < n; i++) {
		if (s[i] == t[0]) {
			dp[i][0] = dp[i - 1][0] + 1;
		}
		else {
			dp[i][0] = dp[i - 1][0];
		}
	}
	for (int i = 1; i < n; i++) {
		for (int j = 1; j < m; j++) {
			if (s[i] == t[j]) {
				dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];

			}
			else {
				dp[i][j] = dp[i - 1][j];
			}
		}
	}

	return dp[n - 1][m - 1];
}

int LClabuladong::countSubstrings(string s)
{
	int ret = 0;
	int n = s.size();
	vector<vector<bool>> dp(n, vector<bool>(n, false));

	for (int i = 0; i < n; i++) {
		int k = 0;
		for (int j = i; j < n; j++) {
			if (k == j && s[k] == s[j]) {
				dp[i][j] = true;
				ret++;
			}
			else if (j - k == 1 && s[k] == s[j]) {
				dp[i][j] = true;
				ret++;
			}
			else if (s[k] == s[j] && dp[i - 2][j - 1]) {
				dp[i][j] = true;
				ret++;
			}
			k++;
		}
	}

	return ret;
}

int LClabuladong::largestRectangleArea(vector<int>& heights)
{
	int res = 0;
	int h, left, right, w;
	heights.push_back(0);
	heights.insert(heights.begin(), 0);
	stack<int> s;
	s.push(0);
	for (int i = 0; i < heights.size(); i++) {
		int temp = heights[i];
		if (temp >= heights[s.top()]) {
			s.push(i);
		}
		else {
			while (temp < heights[s.top()]) {
				h = heights[s.top()];
				s.pop();
				right = i;
				left = s.top();
				w = right - left - 1;
				res = max(res, w * h);
			}
			s.push(i);
		}
	}

	return res;
}

int LClabuladong::minNumber(vector<int>& nums1, vector<int>& nums2)
{
	vector<int> arr(10, 0);
	int c = 10;
	int a = 10, b = 10;
	for (int& i : nums1) {
		arr[i]++;
		a = min(a, i);
	}
	for (int& i : nums2) {
		arr[i]++;
		b = min(b, i);
		if (arr[i] == 2) {
			c = min(c, i);
		}
	}
	if (c != 10) {
		return c;
	}
	int res = (a < b ? a : b) * 10 + (a > b ? a : b);

	return res;
}

TreeNode* LClabuladong::lcaDeepestLeaves(TreeNode* root)
{
	map<int, vector<TreeNode*>> deepNodes;
	lcaDeepestLeaves_CountDeep(root, 0, deepNodes);
	vector<TreeNode*> deepestLeave = deepNodes[deepNodes.size() - 1];
	int n = deepestLeave.size();
	if (n == 1) {
		return deepestLeave[0];
	}
	queue<int> q;
	for (int i = 0; i < n; i++) {
		q.push(deepestLeave[i]->val);
		deepestLeave[i]->val = -1;
	}
	TreeNode* res = NULL;
	lcaDeepestLeaves_FindParents(root, n, res);
	for (int i = 0; i < n; i++) {
		deepestLeave[i]->val = q.front();
		q.pop();
	}

	return res;
}

int LClabuladong::lcaDeepestLeaves_FindParents(TreeNode* root, int n, TreeNode*& res)
{
	if (root->val == -1) {
		return 1;
	}
	int leftNum = 0, rightNum = 0;
	if (root->left) {
		leftNum = lcaDeepestLeaves_FindParents(root->left, n, res);
	}
	if (root->right) {
		rightNum = lcaDeepestLeaves_FindParents(root->right, n, res);
	}
	if (leftNum == -1 || rightNum == -1) {
		return -1;
	}
	if (leftNum + rightNum == n) {
		res = root;
		return -1;
	}
	return leftNum + rightNum;
}

void LClabuladong::lcaDeepestLeaves_CountDeep(TreeNode* root, int d, map<int, vector<TreeNode*>>& deepNodes)
{
	if (deepNodes.count(d)) {
		deepNodes[d].push_back(root);
	}
	else {
		vector<TreeNode*> temp(1, root);
		deepNodes[d] = temp;
	}
	if (root->left) {
		lcaDeepestLeaves_CountDeep(root->left, d + 1, deepNodes);
	}
	if (root->right) {
		lcaDeepestLeaves_CountDeep(root->right, d + 1, deepNodes);
	}
}

long long LClabuladong::repairCars(vector<int>& ranks, int cars)
{
	sort(ranks.begin(), ranks.end(), less<int>());
	long long right = LLONG_MAX;
	long long left = 1;
	while (left < right) {
		long long mid = (right - left) / 2 + left;
		if (repairCars_CanRepair(ranks, cars, mid)) {
			right = mid;
		}
		else {
			left = mid + 1;
		}
	}

	return right;
}

bool LClabuladong::repairCars_CanRepair(vector<int>& ranks, int cars, long long time)
{
	long long okCars = 0;
	for (int& r : ranks) {
		okCars += sqrt(time / r);
		if (okCars >= cars) {
			return true;
		}
	}

	return false;
}

int LClabuladong::findDelayedArrivalTime(int arrivalTime, int delayedTime)
{
	return (arrivalTime + delayedTime) % 24;
}

vector<vector<int>> LClabuladong::queensAttacktheKing(vector<vector<int>>& queens, vector<int>& king)
{
	vector<vector<int>> res(8);//上下左右，左上右上，左下右下
	for (auto& queen : queens) {
		auto temp = queen;
		queen[0] -= king[0];
		queen[1] -= king[1];
		if (queen[0] * queen[1] != 0 && abs(queen[0]) != abs(queen[1])) {
			continue;
		}
		else if (queen[0] == 0 && queen[1] > 0) {
			//右
			if (res[0].size() == 0) {
				res[0] = temp;
				continue;
			}
			res[0][1] = min(res[0][1], temp[1]);
		}
		else if (queen[0] == 0 && queen[1] < 0) {
			//左
			if (res[1].size() == 0) {
				res[1] = temp;
				continue;
			}
			res[1][1] = max(res[1][1], temp[1]);
		}
		else if (queen[1] == 0 && queen[0] > 0) {
			//下
			if (res[2].size() == 0) {
				res[2] = temp;
				continue;
			}
			res[2][0] = min(res[2][0], temp[0]);
		}
		else if (queen[1] == 0 && queen[0] < 0) {
			//上
			if (res[3].size() == 0) {
				res[3] = temp;
				continue;
			}
			res[3][0] = max(res[3][0], temp[0]);
		}
		else if (queen[0] > 0 && queen[1] > 0) {
			//右下
			if (res[4].size() == 0) {
				res[4] = temp;
				continue;
			}
			if (res[4][0] > temp[0]) {
				res[4] = temp;
			}
		}
		else if (queen[0] < 0 && queen[1] < 0) {
			//左上
			if (res[5].size() == 0) {
				res[5] = temp;
				continue;
			}
			if (res[5][0] < temp[0]) {
				res[5] = temp;
			}
		}
		else if (queen[0] > 0 && queen[1] < 0) {
			//右上
			if (res[6].size() == 0) {
				res[6] = temp;
				continue;
			}
			if (res[6][0] > temp[0]) {
				res[6] = temp;
			}
		}
		else if (queen[0] < 0 && queen[1] > 0) {
			//左下
			if (res[7].size() == 0) {
				res[7] = temp;
				continue;
			}
			if (res[7][1] > temp[1]) {
				res[7] = temp;
			}
		}
	}
	vector<vector<int>> res2;
	for (auto& r : res) {
		if (r.size() != 0) {
			res2.push_back(r);
		}
	}

	return res2;
}

int LClabuladong::giveGem(vector<int>& gem, vector<vector<int>>& operations)
{
	for (auto& op : operations) {
		int diff = gem[op[0]] / 2;
		gem[op[0]] -= diff;
		gem[op[1]] += diff;
	}
	int maxN = gem[0];
	int minN = gem[0];

	for (int i = 1; i < gem.size(); i++) {
		maxN = max(maxN, gem[i]);
		minN = min(minN, gem[i]);
	}

	return maxN - minN;
}

vector<vector<int>> LClabuladong::pacificAtlantic(vector<vector<int>>& heights)
{
	vector<vector<int>> result;
	int n = heights.size();
	int m = heights[0].size(); // 这里不用担心空指针，题目要求说了长宽都大于1

	// 记录从太平洋边出发，可以遍历的节点
	vector<vector<bool>> pacific = vector<vector<bool>>(n, vector<bool>(m, false));

	// 记录从大西洋出发，可以遍历的节点
	vector<vector<bool>> atlantic = vector<vector<bool>>(n, vector<bool>(m, false));

	// 从最上最下行的节点出发，向高处遍历
	for (int i = 0; i < n; i++) {
		pacificAtlantic_dfs(heights, pacific, i, 0); // 遍历最上行，接触太平洋
		pacificAtlantic_dfs(heights, atlantic, i, m - 1); // 遍历最下行，接触大西洋
	}

	// 从最左最右列的节点出发，向高处遍历
	for (int j = 0; j < m; j++) {
		pacificAtlantic_dfs(heights, pacific, 0, j); // 遍历最左列，接触太平洋
		pacificAtlantic_dfs(heights, atlantic, n - 1, j); // 遍历最右列，接触大西洋
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			// 如果这个节点，从太平洋和大西洋出发都遍历过，就是结果
			if (pacific[i][j] && atlantic[i][j]) result.push_back({ i, j });
		}
	}
	return result;
}

void LClabuladong::pacificAtlantic_dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited, int x, int y)
{
	if (visited[x][y]) return;
	visited[x][y] = true;
	for (int i = 0; i < 4; i++) { // 向四个方向遍历
		int nextx = x + dir[i][0];
		int nexty = y + dir[i][1];
		// 超过边界
		if (nextx < 0 || nextx >= heights.size() || nexty < 0 || nexty >= heights[0].size()) continue;
		// 高度不合适，注意这里是从低向高判断
		if (heights[x][y] > heights[nextx][nexty]) continue;

		pacificAtlantic_dfs(heights, visited, nextx, nexty);
	}
	return;;
}

int LClabuladong::minCount(vector<int>& coins)
{
	int res = 0;
	for (int& coin : coins) {
		res += ((coin + 1) / 2);
	}

	return res;
}

int LClabuladong::collectTheCoins(vector<int>& coins, vector<vector<int>>& edges)
{
	int n = coins.size();
	vector<vector<int>> g(n);
	vector<int> degree(n);
	for (const auto& edge : edges) {
		int x = edge[0], y = edge[1];
		g[x].push_back(y);
		g[y].push_back(x);
		++degree[x];
		++degree[y];
	}

	int rest = n;
	{
		/* 删除树中所有无金币的叶子节点，直到树中所有的叶子节点都是含有金币的 */
		queue<int> q;
		for (int i = 0; i < n; ++i) {
			if (degree[i] == 1 && !coins[i]) {
				q.push(i);
			}
		}
		while (!q.empty()) {
			int u = q.front();
			--degree[u];
			q.pop();
			--rest;
			for (int v : g[u]) {
				--degree[v];
				if (degree[v] == 1 && !coins[v]) {
					q.push(v);
				}
			}
		}
	}
	{
		/* 删除树中所有的叶子节点, 连续删除2次 */
		for (int _ = 0; _ < 2; ++_) {
			queue<int> q;
			for (int i = 0; i < n; ++i) {
				if (degree[i] == 1) {
					q.push(i);
				}
			}
			while (!q.empty()) {
				int u = q.front();
				--degree[u];
				q.pop();
				--rest;
				for (int v : g[u]) {
					--degree[v];
				}
			}
		}
	}

	return rest == 0 ? 0 : (rest - 1) * 2;
}

int LClabuladong::distMoney(int money, int children)
{
	if (money < children) {
		return -1;
	}
	money -= children;
	int res = money / 7;
	if (res > children) {
		return children - 1;
	}
	if (res == children) {
		if (money % 7 == 0) {
			return res;
		}
		else {
			return max(0, res - 1);
		}
	}
	if (money % 7 == 3 && children - res == 1) {
		return max(0, res - 1);
	}

	return res;
}

bool LClabuladong::makeStringsEqual(string s, string target)
{
	if (s == target) {
		return true;
	}
	bool sHas1 = false;
	bool targetHas1 = false;
	for (int i = 0; i < s.size(); i++) {
		sHas1 = (s[i] == '1' || sHas1);
		targetHas1 = (target[i] == '1' || targetHas1);
	}

	return sHas1 && targetHas1;
}

long long LClabuladong::maxKelements(vector<int>& nums, int k)
{
	priority_queue<int, vector<int>, less<int>> pq1(nums.begin(), nums.end());
	long long ans = 0;
	while (k--) {
		int t = pq1.top();
		pq1.pop();
		ans += t;
		t = ceil(double(t) / 3);
		pq1.push(t);
	}

	return ans;
}

int LClabuladong::beautifulSubsets(vector<int>& nums, int k)
{
	unordered_map<int, vector<int>> classes;
	for (int e : nums)
		classes[e % k].push_back(e);
	int ans = 1;
	for (auto& [_, v] : classes) {
		sort(v.begin(), v.end());
		const int l = v.size();
		int dp1 = 1, dp2 = 1;
		int pre = INT_MIN;
		for (int i = 0; i < l;) {
			const int j = i;
			do {
				++i;
			} while (i < l && nums[i] == nums[j]);
			const int t = v[j] != pre + k ? dp1 : dp2;
			const int dp0 = dp1 + (t << (i - j)) - t;
			dp2 = dp1;
			dp1 = dp0;
			pre = v[j];
		}
		ans *= dp1;
	}
	return ans - 1;
}

vector<int> LClabuladong::singleNumberIII(vector<int>& nums)
{
	int xorsum = 0;
	for (int num : nums) {
		xorsum ^= num;
	}
	bitset<sizeof(int) * 8> bXorsum(xorsum);
	int index = 0;
	for (int i = 0; i < bXorsum.size(); i++) {
		if (bXorsum[i] == 1) {
			index = i;
			break;
		}
	}
	bitset<bXorsum.size()> t(0);
	t[index] = 1;
	int lsb = t.to_ullong();
	int type1 = 0, type2 = 0;
	for (int num : nums) {
		if (num & lsb) {
			type1 ^= num;
		}
		else {
			type2 ^= num;
		}
	}
	return { type1, type2 };
}

int LClabuladong::sumOfMultiples(int n)
{
	int ret = 0;
	for (int i = 1; i <= n; i++) {
		if (i % 3 == 0 || i % 5 == 0 || i % 7 == 0) {
			ret += i;
		}
	}

	return ret;
}

vector<int> LClabuladong::rowAndMaximumOnes(vector<vector<int>>& mat)
{
	vector<int> ans(2, -1);
	for (int i = 0; i < mat.size(); i++) {
		int oneNum = 0;
		for (int j = 0; j < mat[i].size(); j++) {
			if (mat[i][j] == 1) {
				oneNum += 1;
			}
		}
		if (oneNum > ans[1]) {
			ans[0] = i;
			ans[1] = oneNum;
		}
	}

	return ans;
}

int LClabuladong::miceAndCheese(vector<int>& reward1, vector<int>& reward2, int k)
{
	int ans = 0;
	vector<int> rw3(reward1.size());
	for (int i = 0; i < rw3.size(); i++) {
		rw3[i] = reward1[i] - reward2[i];
		ans += reward2[i];
	}
	sort(rw3.begin(), rw3.end(), greater<int>());
	for (int i = 0; i < k; i++) {
		ans += rw3[i];
	}

	return ans;
}

int LClabuladong::alternateDigitSum(int n)
{
	int ans = 0;
	int count = 0;
	int flag = 1;
	while (n) {
		ans += (flag * (n % 10));
		flag *= -1;
		n /= 10;
		count++;
	}
	ans = ans * (count % 2 == 0 ? -1 : 1);

	return ans;
}

int LClabuladong::minScore(int n, vector<vector<int>>& roads)
{
	int ans = 0;
	vector<list<int>> mmap(n + 1, list<int>());
	for (int i = 0; i < roads.size(); i++) {
		mmap[roads[i][0]].push_back(roads[i][1]);
		mmap[roads[i][1]].push_back(roads[i][0]);
	}
	unordered_set<int> unionMap;
	unionMap.insert(1);
	for (list<int>::iterator i = mmap[1].begin(); i != mmap[1].end(); i++) {
		if (!unionMap.count(*i)) {
			unionMap.insert(*i);
			minScore_dfs(unionMap, mmap, *i);
		}
	}
	auto cmp = [](vector<int>& a, vector<int>& b)->bool {
		return a[2] < b[2];
		};
	sort(roads.begin(), roads.end(), cmp);
	for (int i = 0; i < roads.size(); i++) {
		if (unionMap.count(roads[i][0])) {
			ans = roads[i][2];
			break;
		}
	}

	return ans;
}

void LClabuladong::minScore_dfs(unordered_set<int>& unionMap, vector<list<int>>& mapp, int n)
{
	for (list<int>::iterator i = mapp[n].begin(); i != mapp[n].end(); i++) {
		if (!unionMap.count(*i)) {
			unionMap.insert(*i);
			minScore_dfs(unionMap, mapp, *i);
		}
	}
}

int LClabuladong::tupleSameProduct(vector<int>& nums)
{
	unordered_map<int, int> value_pair;
	vector<int> curPair;
	int ans = 0;
	for (int i = 0; i < nums.size(); i++) {
		curPair.push_back(nums[i]);
		tupleSameProduct_Recursion(nums, value_pair, curPair, i + 1);
		curPair.pop_back();
	}
	for (auto& p : value_pair) {
		if (p.second < 2) {
			continue;
		}
		ans += ((p.second * (p.second - 1) / 2) * 8);
	}

	return ans;
}

void LClabuladong::tupleSameProduct_Recursion(vector<int>& nums, unordered_map<int, int>& value_pair, vector<int>& curPair, int n)
{
	if (curPair.size() == 2) {
		int t = curPair[0] * curPair[1];
		if (value_pair.count(t)) {
			value_pair[t]++;
		}
		else {
			value_pair[t] = 1;
		}
		return;
	}

	for (int i = n; i < nums.size(); i++) {
		curPair.push_back(nums[i]);
		tupleSameProduct_Recursion(nums, value_pair, curPair, i + 1);
		curPair.pop_back();
	}
}

int LClabuladong::minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships)
{
	auto canTalk = [](vector<vector<int>>& languages, int a, int b)->bool {
		for (int i = 0; i < languages[a - 1].size(); i++) {
			for (int j = 0; j < languages[b - 1].size(); j++) {
				if (languages[a - 1][i] == languages[b - 1][j]) return true;
			}
		}
		return false;
		};
	vector<bool> talkFlag(languages.size(), false);
	for (int i = 0; i < friendships.size(); i++) {
		if (!canTalk(languages, friendships[i][0], friendships[i][1])) {
			talkFlag[friendships[i][0] - 1] = true;
			talkFlag[friendships[i][1] - 1] = true;
		}
	}
	int cnt = 0;
	vector<int> lanCnt(n, 0);
	for (int i = 0; i < talkFlag.size(); i++) {
		if (talkFlag[i]) {
			cnt++;
			for (int j = 0; j < languages[i].size(); j++) {
				lanCnt[languages[i][j] - 1]++;
			}
		}
	}
	int maxIndex = *max_element(lanCnt.begin(), lanCnt.end());
	int ans = cnt - maxIndex;

	return ans;
}

string LClabuladong::categorizeBox(int length, int width, int height, int mass)
{
	string ans[2] = { "Heavy", "Bulky" };
	bool flag[2] = { false };
	if (length >= 1e4 || width >= 1e4 || height >= 1e4 || long(length) * long(width) * long(height) >= 1e9) {
		flag[1] = true;
	}
	flag[0] = (mass >= 100);

	if (flag[0] != flag[1]) {
		if (flag[0]) {
			return ans[0];
		}
		else {
			return ans[1];
		}
	}

	return flag[0] ? string("Both") : string("Neither");
}

bool LClabuladong::checkValidGrid(vector<vector<int>>& grid)
{
	int len = grid.size();
	auto Step = [](vector<vector<int>>& grid, int x, int y, int& n)->pair<int, int> {
		if (x - 2 >= 0 && y - 1 >= 0 && grid[x - 2][y - 1] == n + 1) {
			return pair<int, int>(x - 2, y - 1);
		}
		else if (x + 2 < grid.size() && y - 1 >= 0 && grid[x + 2][y - 1] == n + 1) {
			return pair<int, int>(x + 2, y - 1);
		}
		else if (x + 2 < grid.size() && y + 1 < grid.size() && grid[x + 2][y + 1] == n + 1) {
			return pair<int, int>(x + 2, y + 1);
		}
		else if (x - 2 >= 0 && y + 1 < grid.size() && grid[x - 2][y + 1] == n + 1) {
			return pair<int, int>(x - 2, y + 1);
		}
		else if (x - 1 >= 0 && y - 2 >= 0 && grid[x - 1][y - 2] == n + 1) {
			return pair<int, int>(x - 1, y - 2);
		}
		else if (x + 1 < grid.size() && y - 2 >= 0 && grid[x + 1][y - 2] == n + 1) {
			return pair<int, int>(x + 1, y - 2);
		}
		else if (x + 1 < grid.size() && y + 2 < grid.size() && grid[x + 1][y + 2] == n + 1) {
			return pair<int, int>(x + 1, y + 2);
		}
		else if (x - 1 >= 0 && y + 2 < grid.size() && grid[x - 1][y + 2] == n + 1) {
			return pair<int, int>(x - 1, y + 2);
		}

		return pair<int, int>(-1, -1);
		};
	pair<int, int> curPos(0, 0);
	int n;
	if (grid[0][0] == 0) {
		n = 0;
	}
	else {
		n = -1;
	}

	while (curPos.first != -1) {
		curPos = Step(grid, curPos.first, curPos.second, n);
		if (curPos.first != -1) {
			n += 1;
		}
	}
	if (n == len * len - 1) {
		return true;
	}

	return false;
}

long long LClabuladong::countPairs(int n, vector<vector<int>>& edges)
{
	class UnionFind
	{
	public:
		int count;
		std::vector<int> parent;
		std::vector<int> size;
	public:
		UnionFind(int n) {
			this->count = n;
			parent.resize(n);
			size.resize(n, 1);
			for (int i = 0; i < parent.size(); i++) {
				parent[i] = i;
			}
		}
		//将p q连通
		void union_(int p, int q) {
			int rootP = findRoot(p);
			int rootQ = findRoot(q);
			if (rootP == rootQ) {
				return;
			}
			if (size[rootP] > size[rootQ]) {
				parent[rootQ] = rootP;
				size[rootP] += size[rootQ];
			}
			else {
				parent[rootP] = rootQ;
				size[rootQ] += size[rootP];
			}
			count--;
		}
		//判断p  q是否连通
		bool isConnected(int p, int q) {
			int rootP = findRoot(p);
			int rootQ = findRoot(q);
			return rootP == rootQ;
		}
		//寻找x的根节点
		int findRoot(int x) {
			while (parent[x] != x) {
				parent[x] = parent[parent[x]];
				x = parent[x];
			}
			return x;
		}
		//返回连通分量个数
		int count_() {
			return this->count;
		}
	};
	UnionFind uf(n);
	for (int i = 0; i < edges.size(); i++) {
		uf.union_(edges[i][0], edges[i][1]);
	}
	vector<long long> unionSize;
	long long ret = 0;
	for (int i = 0; i < uf.parent.size(); i++) {
		if (uf.parent[i] == i) {
			unionSize.push_back(uf.size[i]);
		}
	}
	int t = n;
	for (int i = 0; i < unionSize.size(); i++) {
		t -= unionSize[i];
		ret += (unionSize[i] * t);
	}

	return ret;
}

int LClabuladong::countSeniors(vector<string>& details)
{
	int ret = 0;
	for (int i = 0; i < details.size(); i++) {
		int old = (atoi(details[i].substr(11, 2).c_str()));
		if (old > 60) {
			ret++;
		}
	}

	return ret;
}

int LClabuladong::findValueOfPartition(vector<int>& nums)
{
	int maxNums1, minNums2, ans = INT_MAX;
	sort(nums.begin(), nums.end());
	for (int i = 1; i < nums.size(); i++) {
		if (abs(nums[i] - nums[i - 1]) < ans) {
			maxNums1 = min(nums[i], nums[i - 1]);
			minNums2 = max(nums[i], nums[i - 1]);
			ans = abs(maxNums1 - minNums2);
		}
	}
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] > maxNums1 && nums[i] < minNums2) {
			int a = abs(nums[i] - maxNums1);
			int b = abs(nums[i] - minNums2);
			if (a < b) {
				maxNums1 = nums[i];
			}
			else {
				minNums2 = nums[i];
			}
			ans += (min(a, b));
		}
	}

	return ans;
}

int LClabuladong::numRollsToTarget(int n, int k, int target)
{
	int ret = 0;
	vector<vector<int>> memo(n + 1, vector<int>(target + 1, 0));
	for (int i = 1; i < target + 1; i++) {
		if (i <= k) {
			memo[1][i] = 1;
		}
	}
	for (int i = 2; i < n + 1; i++) {
		for (int j = 1; j < target + 1; j++) {
			for (int t = j - 1; t >= 0; t--) {
				if (t + k < j) {
					break;
				}
				memo[i][j] = (memo[i][j] + memo[i - 1][t]) % 1000000007;
			}
		}
	}

	return memo[n][target];
}

vector<int> LClabuladong::distributeCandies(int candies, int num_people)
{
	vector<int> ret(num_people, 0);
	int peopleIndex = 0;
	int curDisNum = 1;
	while (candies > 0) {
		if (curDisNum >= candies) {
			ret[peopleIndex] += candies;
			break;
		}
		ret[peopleIndex] += curDisNum;
		candies -= curDisNum;
		peopleIndex = (peopleIndex + 1) % num_people;
		curDisNum++;
	}

	return ret;
}

int LClabuladong::punishmentNumber(int n)
{
	int ret = 0;

	for (int i = 1; i <= n; i++) {
		int square = i * i;
		string strSquare = to_string(square);
		if (punishmentNumber_Recursion(i, strSquare, 0, 0)) {
			ret += (i * i);
		}
	}

	return ret;
}

bool LClabuladong::punishmentNumber_Recursion(int num, string strNum, int index, int sum)
{
	if (index == strNum.size()) {
		return num == sum;
	}

	for (int i = index; i < strNum.size(); i++) {
		sum += (stoi(strNum.substr(index, i - index + 1)));
		if (sum > num) {
			return false;
		}
		if (punishmentNumber_Recursion(num, strNum, i + 1, sum)) {
			return true;
		}
		sum -= (stoi(strNum.substr(index, i - index + 1)));
	}
	return false;
}

int LClabuladong::makeTheIntegerZero(int num1, int num2)
{
	auto canCount = [=](long long target, int cnt)->bool {
		bitset<64> bTarget;
		while (target) {
			bTarget = bTarget << 1;
			if (target % 2 == 1) {
				bTarget[0] = 1;
			}
			target = target / 2;
		}
		return bTarget.count() <= cnt;
		};
	for (long long i = 1; i < 50; i++) {
		long long target = num1 - i * num2;
		if (target < i) {
			if (num2 >= 0) {
				return -1;
			}
			continue;
		}
		if (canCount(target, i)) {
			return i;
		}
	}

	return -1;
}

int LClabuladong::countDigits(int num)
{
	int ret = 0;
	string strNum = to_string(num);
	for (int i = 0; i < strNum.size(); i++) {
		if (num % (strNum[i] - '0') == 0) {
			ret++;
		}
	}

	return ret;
}

int LClabuladong::rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k)
{
	vector<vector<int>> g(edges.size() + 1);
	for (auto& e : edges) {
		int x = e[0], y = e[1];
		g[x].push_back(y);
		g[y].push_back(x);
	}

	unordered_set<long> s;
	for (auto& e : guesses) {
		s.insert((long)e[0] << 32 | e[1]);
	}

	int ans = 0, cnt0 = 0;
	function<void(int, int)> dfs = [&](int x, int fa) {
		for (int y : g[x]) {
			if (y != fa) {
				cnt0 += s.count((long)x << 32 | y);
				dfs(y, x);
			}
		}
		};
	dfs(0, -1);

	function<void(int, int, int)> reroot = [&](int x, int fa, int cnt) {
		ans += cnt >= k;
		for (int y : g[x]) {
			if (y != fa) {
				reroot(y, x, cnt
					- s.count((long)x << 32 | y) // 原来是对的，现在错了
					+ s.count((long)y << 32 | x)); // 原来是错的，现在对了
			}
		}
		};
	reroot(0, -1, cnt0);
	return ans;
}

int LClabuladong::maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts)
{
	sort(horizontalCuts.begin(), horizontalCuts.end(), less<int>());
	sort(verticalCuts.begin(), verticalCuts.end(), less<int>());
	horizontalCuts.push_back(h);
	verticalCuts.push_back(w);
	int maxH = horizontalCuts[0], maxW = verticalCuts[0];
	for (int i = 1; i < horizontalCuts.size(); i++) {
		maxH = max(maxH, horizontalCuts[i] - horizontalCuts[i - 1]);
	}
	for (int i = 1; i < verticalCuts.size(); i++) {
		maxW = max(maxW, verticalCuts[i] - verticalCuts[i - 1]);
	}
	return ((long long)maxH * (long long)maxW) % 1000000007;
}

long long LClabuladong::pickGifts(vector<int>& gifts, int k)
{
	long long sum = 0;
	long long out = 0;
	priority_queue<int, vector<int>, less<int>> giftStack;
	for (int i = 0; i < gifts.size(); i++) {
		sum += gifts[i];
		giftStack.push(gifts[i]);
	}
	while (k--) {
		int t = giftStack.top();
		giftStack.pop();
		out += (t - int(sqrt(t)));
		giftStack.push(sqrt(t));
	}

	return sum - out;
}

int LClabuladong::minimizeMax(vector<int>& nums, int p)
{
	sort(nums.begin(), nums.end());
	int left = -1, right = nums.back() - nums[0]; // 开区间
	while (left + 1 < right) { // 开区间
		int mid = left + (right - left) / 2, cnt = 0;
		for (int i = 0; i < nums.size() - 1; ++i)
			if (nums[i + 1] - nums[i] <= mid) { // 都选
				++cnt;
				++i;
			}
		(cnt >= p ? right : left) = mid;
	}
	return right;
}

int LClabuladong::hIndex(vector<int>& citations)
{
	int n = citations.size();
	int ret = 0;
	int left = 0, right = n;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (mid == 0) {
			left = mid + 1;
			continue;
		}
		if (citations[n - mid] >= mid) {
			ret = mid;
			left = mid + 1;
			continue;
		}
		right = mid - 1;
	}

	return ret;
}

vector<int> LClabuladong::smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums)
{
	int n = parents.size();
	vector<vector<int>> children(n);
	for (int i = 1; i < n; i++) {
		children[parents[i]].push_back(i);
	}

	vector<int> res(n, 1);
	vector<unordered_set<int>> geneSet(n);
	function<int(int)> dfs = [&](int node) -> int {
		geneSet[node].insert(nums[node]);
		for (auto child : children[node]) {
			res[node] = max(res[node], dfs(child));
			if (geneSet[node].size() < geneSet[child].size()) {
				geneSet[node].swap(geneSet[child]);
			}
			geneSet[node].merge(geneSet[child]);
		}
		while (geneSet[node].count(res[node]) > 0) {
			res[node]++;
		}
		return res[node];
		};
	dfs(0);
	return res;
}

Node2* LClabuladong::connect(Node2* root)
{
	list<Node2*> left1, right1, left2, right2;
	if (root->left) {
		connect_Recursion(root->left, left1, right1);
	}
	if (root->right) {
		connect_Recursion(root->right, left2, right2);
	}
	while (!right1.empty() && !left2.empty()) {
		Node2* pnLeft = right1.front();
		right1.pop_front();
		Node2* pnRight = left2.front();
		left2.pop_front();
		pnLeft->next = pnRight;
	}
	return root;
}

void LClabuladong::connect_Recursion(Node2* root, list<Node2*>& left, list<Node2*>& right)
{
	list<Node2*> left1, right1, left2, right2;
	if (root->left) {
		connect_Recursion(root->left, left1, right1);
	}
	if (root->right) {
		connect_Recursion(root->right, left2, right2);
	}
	while (!right1.empty() && !left2.empty()) {
		Node2* pnLeft = right1.front();
		right1.pop_front();
		Node2* pnRight = left2.front();
		left2.pop_front();
		pnLeft->next = pnRight;
	}
	while (!right1.empty()) {
		right2.push_back(right1.front());
		right1.pop_front();
	}
	while (!left2.empty()) {
		left1.push_back(left2.front());
		left2.pop_front();
	}
	left1.push_front(root);
	right2.push_front(root);
	left = left1;
	right = right2;
}

bool LClabuladong::containsDuplicate(vector<int>& nums)
{
	unordered_set<int> mm;
	for (int i = 0; i < nums.size(); i++) {
		if (mm.count(nums[i])) {
			return true;
		}
		mm.insert(nums[i]);
	}
	return false;
}

int LClabuladong::findTheLongestBalancedSubstring(string s)
{
	int ret = 0;
	int nZero = 0, nOne = 0;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '0') {
			if (i != 0 && s[i - 1] == '1') {
				if (nZero <= nOne) {
					ret = max(ret, nZero * 2);
				}
				else if (nZero > nOne) {
					ret = max(ret, 2 * nOne);
				}
				nZero = 1;
				nOne = 0;
				continue;
			}
			nZero++;
		}
		else {
			nOne++;
		}
	}
	if (nZero <= nOne) {
		ret = max(ret, nZero * 2);
	}
	else if (nZero > nOne) {
		ret = max(ret, 2 * nOne);
	}

	return ret;
}

long long LClabuladong::kthLargestLevelSum(TreeNode* root, int k)
{
	list<long long> lst = kthLargestLevelSum_Recursion(root);
	if (lst.size() < k) {
		return -1;
	}
	lst.sort();
	lst.reverse();
	while (--k) {
		lst.pop_front();
	}

	return lst.front();
}

list<long long> LClabuladong::kthLargestLevelSum_Recursion(TreeNode* root)
{
	list<long long> leftLst;
	list<long long> rightLst;
	if (root->left)
		leftLst = kthLargestLevelSum_Recursion(root->left);
	if (root->right)
		rightLst = kthLargestLevelSum_Recursion(root->right);
	list<long long> ret;
	while (!leftLst.empty() && !rightLst.empty()) {
		ret.push_back(leftLst.front() + rightLst.front());
		leftLst.pop_front();
		rightLst.pop_front();
	}
	if (!leftLst.empty()) {
		ret.splice(ret.end(), leftLst);
	}
	if (!rightLst.empty()) {
		ret.splice(ret.end(), rightLst);
	}
	ret.push_front(root->val);
	return ret;
}

int LClabuladong::maximumMinutes(vector<vector<int>>& grid)
{
	int length = grid.size(), width = grid[0].size();
	vector<vector<int>> orgGrid = grid;
	list<pair<int, int>> lstFire;
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			if (grid[i][j] == 1) {
				lstFire.push_back(pair(i, j));
				grid[i][j] = 0;
				continue;
			}
			else if (grid[i][j] == 0) {
				grid[i][j] = -1;
			}
			else if (grid[i][j] == 2) {
				grid[i][j] = -2;
			}
		}
	}
	while (!lstFire.empty()) {
		int x = lstFire.front().first;
		int y = lstFire.front().second;
		lstFire.pop_front();
		int time = grid[x][y] + 1;
		if (x + 1 < length && (grid[x + 1][y] == -1 || grid[x + 1][y] > time)) {
			grid[x + 1][y] = time;
			lstFire.push_back(pair(x + 1, y));
		}

		if (x - 1 >= 0 && (grid[x - 1][y] == -1 || grid[x - 1][y] > time)) {
			grid[x - 1][y] = time;
			lstFire.push_back(pair(x - 1, y));
		}

		if (y + 1 < width && (grid[x][y + 1] == -1 || grid[x][y + 1] > time)) {
			grid[x][y + 1] = time;
			lstFire.push_back(pair(x, y + 1));
		}

		if (y - 1 >= 0 && (grid[x][y - 1] == -1 || grid[x][y - 1] > time)) {
			grid[x][y - 1] = time;
			lstFire.push_back(pair(x, y - 1));
		}
	}

	list<pair<int, int>> lstStep;
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			if (orgGrid[i][j] == 1) {
				orgGrid[i][j] = -1;
			}
			else if (orgGrid[i][j] == 0) {
				orgGrid[i][j] = -1;
			}
			else if (orgGrid[i][j] == 2) {
				orgGrid[i][j] = -2;
			}
		}
	}

	orgGrid[0][0] = 0;
	lstStep.push_back(pair(0, 0));
	while (!lstStep.empty()) {
		int x = lstStep.front().first;
		int y = lstStep.front().second;
		lstStep.pop_front();
		int time = orgGrid[x][y] + 1;
		if (x + 1 < length && (time < orgGrid[x + 1][y] || orgGrid[x + 1][y] == -1)) {
			orgGrid[x + 1][y] = time;
			lstStep.push_back(pair(x + 1, y));
		}

		if (x - 1 >= 0 && (orgGrid[x - 1][y] == -1 || orgGrid[x - 1][y] > time)) {
			orgGrid[x - 1][y] = time;
			lstStep.push_back(pair(x - 1, y));
		}

		if (y + 1 < width && (orgGrid[x][y + 1] == -1 || orgGrid[x][y + 1] > time)) {
			orgGrid[x][y + 1] = time;
			lstStep.push_back(pair(x, y + 1));
		}

		if (y - 1 >= 0 && (orgGrid[x][y - 1] == -1 || orgGrid[x][y - 1] > time)) {
			orgGrid[x][y - 1] = time;
			lstStep.push_back(pair(x, y - 1));
		}
	}

	vector<vector<int>> stepGrid(length, vector<int>(width));
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < width; j++) {
			if (grid[i][j] == -1) {
				stepGrid[i][j] = 1000000000;
				continue;
			}
			if (grid[i][j] == -2) {
				stepGrid[i][j] = -1;
				continue;
			}
			stepGrid[i][j] = grid[i][j] - orgGrid[i][j] - 1;
		}
	}
	maximumMinutes_ret = -1;
	stepGrid[length - 1][width - 1] += 1;
	vector<vector<bool>> isLook(length, vector<bool>(width, false));
	maximumMinutes_Recursion(stepGrid, isLook, 0, 0, 1000000000);


	return maximumMinutes_ret;
}

void LClabuladong::maximumMinutes_Recursion(vector<vector<int>>& grid, vector<vector<bool>>& isLook, int x, int y, int curMin)
{
	int length = grid.size(), width = grid[0].size();
	if (x == length - 1 && y == width - 1) {
		maximumMinutes_ret = curMin;

		return;
	}
	if (x + 1 < length && !isLook[x + 1][y] && grid[x + 1][y] >= 0 && min(curMin, grid[x + 1][y]) > maximumMinutes_ret) {
		isLook[x + 1][y] = true;
		maximumMinutes_Recursion(grid, isLook, x + 1, y, min(curMin, grid[x + 1][y]));
		isLook[x + 1][y] = false;
	}

	if (x - 1 >= 0 && !isLook[x - 1][y] && grid[x - 1][y] >= 0 && min(curMin, grid[x - 1][y]) > maximumMinutes_ret) {
		isLook[x - 1][y] = true;
		maximumMinutes_Recursion(grid, isLook, x - 1, y, min(curMin, grid[x - 1][y]));
		isLook[x - 1][y] = false;
	}

	if (y + 1 < width && !isLook[x][y + 1] && grid[x][y + 1] >= 0 && min(curMin, grid[x][y + 1]) > maximumMinutes_ret) {
		isLook[x][y + 1] = true;
		maximumMinutes_Recursion(grid, isLook, x, y + 1, min(curMin, grid[x][y + 1]));
		isLook[x][y + 1] = false;
	}

	if (y - 1 >= 0 && !isLook[x][y - 1] && grid[x][y - 1] >= 0 && min(curMin, grid[x][y - 1]) > maximumMinutes_ret) {
		isLook[x][y - 1] = true;
		maximumMinutes_Recursion(grid, isLook, x, y - 1, min(curMin, grid[x][y - 1]));
		isLook[x][y - 1] = false;
	}
}

vector<int> LClabuladong::successfulPairs(vector<int>& spells, vector<int>& potions, long long success)
{
	int n = spells.size(), m = potions.size();
	sort(potions.begin(), potions.end());
	vector<int> ret(n, 0);
	for (int i = 0; i < n; i++) {
		long long tSpell = spells[i];
		int left = 0, right = potions.size();
		while (left < right) {
			int mid = left + (right - left) / 2;
			if (tSpell * potions[mid] >= success) {
				right = mid;
			}
			else {
				left = mid+1;//循环的出口一定是这个，因为mid最多只会等于left，不会等于right
			}
		}
		ret[i] = m - right;
	}

	return ret;
}

int LClabuladong::findPeakElement(vector<int>& nums)
{
	int n = nums.size();
	int left = 0, right = n - 1;
	while (left < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] > nums[mid + 1]) {
			right = mid;
		}
		else {
			left = mid + 1;
		}
	}

	return left;
}

int LClabuladong::maximumRows(vector<vector<int>>& matrix, int numSelect)
{
	int m, n;
	m = matrix.size();
	n = matrix[0].size();
	vector<int> mask(m, 0);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			mask[i] += matrix[i][j] << (n - j - 1);
		}
	}
	int cur = 0;
	int res = 0;
	int limit = (1 << n);
	while (cur < limit) {
		if (__popcnt(cur) != numSelect) {
			cur++;
			continue;
		}
		int t = 0;
		for (int i = 0; i < m; i++) {
			if ((mask[i] & cur) == mask[i]) {
				t++;
			}
		}
		res = max(res, t);
		cur++;
	}

	return res;
}

vector<int> LClabuladong::canSeePersonsCount(vector<int>& heights)
{
	int n = heights.size();
	vector<int> res(n, 0);
	list<int> stack;
	stack.push_back(0);
	for (int i = 1; i < n; i++) {
		if (stack.empty()) {
			stack.push_back(i);
			continue;
		}
		int lastNum = heights[stack.back()];
		res[stack.back()]++;
		if (heights[i] >= lastNum) {
			stack.pop_back();
			i--;
			continue;
		}
		stack.push_back(i);
	}

	return res;
}

ListNode* LClabuladong::insertGreatestCommonDivisors(ListNode* head)
{
	auto CommonDivisor = [](int a, int b)->int {
		int q = max(a, b), p = max(a, b);
		int temp = min(a, b);
		while (temp != 0) {
			q = p;
			p = temp;
			temp = q % p;
		}
		return p;
	};
	ListNode* pa,* pb;
	pa = head;
	pb = head->next;
	while (pa != NULL && pb != NULL) {
		int t = CommonDivisor(pa->val, pb->val);
		ListNode* temp = new ListNode();
		temp->val = t;
		temp->next = pb;
		pa->next = temp;
		pa = pb;
		pb = pa->next;
	}

	return head;
}

int LClabuladong::numberOfBoomerangs(vector<vector<int>>& points)
{
	auto EuclideamDistance = [&](vector<int>& a, vector<int>& b)->int {
		return pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[1]), 2);
		};
	int n = points.size();
	vector<unordered_map<int, int>> mm(n);
	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
			int dis = EuclideamDistance(points[i], points[j]);
			mm[i][dis]++;
			mm[j][dis]++;
		}
	}
	int res = 0;
	for (int i = 0; i < n; i++) {
		unordered_map<int, int>& tm = mm[i];
		for (unordered_map<int, int>::iterator it = tm.begin(); it != tm.end(); it++) {
			res += it->second*(it->second-1);
		}
	}

	return res;
}

int LClabuladong::minExtraChar(string s, vector<string>& dictionary)
{
	int n = dictionary.size();
	unordered_map<string, int> mp;
	for (int i = 0; i < n; i++) {
		mp[dictionary[i]]++;
	}
	vector<int> dp(s.size() + 1, INT_MAX);
	dp[0] = 0;
	for (int i = 1; i < dp.size(); i++) {
		dp[i] = dp[i - 1] + 1;
		for (int j = i - 1; j >= 0; j--) {
			if (mp.count(s.substr(j, i - j))) {
				if (j == 0) {
					dp[i] = 0;
					continue;
				}
				dp[i] = min(dp[i], dp[j]);
			}
		}
	}

	return dp[s.size()];
}

int LClabuladong::minLength(string s)
{
	stack<char> ss;
	for (int i = 0; i < s.size(); i++) {
		if (ss.empty()|| (s[i] != 'B' && s[i] != 'D')) {
			ss.push(s[i]);
			continue;
		}
		if (s[i] - ss.top() == 1) {
			ss.pop();
		}
		else {
			ss.push(s[i]);
		}
	}

	return ss.size();
}

int LClabuladong::addMinimum(string word)
{
	int n = word.size();
	vector<int> dp(n + 1, 0);
	dp[1] = 2;
	for (int i = 2; i < n + 1; i++) {
		if (word[i - 1] > word[i - 2]) {
			dp[i] = dp[i - 1] - 1;
		}
		else {
			dp[i] = dp[i - 1] + 2;
		}
	}
	return dp[n];
}

int LClabuladong::countWords(vector<string>& words1, vector<string>& words2)
{
	unordered_map<string, int> freq1, freq2;
	for (const string& w : words1) {
		++freq1[w];
	}
	for (const string& w : words2) {
		++freq2[w];
	}

	int res = 0;
	for (const auto& [w, cnt1] : freq1) {
		if (cnt1 == 1 && freq2[w] == 1) {
			++res;
		}
	}
	return res;
}

ListNode* LClabuladong::deleteDuplicates(ListNode* head)
{
	if (head == NULL) {
		return NULL;
	}
	ListNode* h = new ListNode();
	h->next = head;
	ListNode* lp, * rp;
	lp = h;
	rp = h->next;
	while (rp) {
		if (rp->next == NULL) {
			break;
		}
		if (rp->val == rp->next->val) {
			int val = rp->val;
			while (rp != NULL && rp->val == val) {
				lp->next = rp->next;
				delete rp;
				rp = lp->next;
			}
			continue;
		}
		lp = rp;
		rp = lp->next;
	}

	return h->next;
}

int LClabuladong::count(string num1, string num2, int min_sum, int max_sum)
{
	return 0;
}

int LClabuladong::countSpecialNumbers(int n)
{
	string nStr = to_string(n);
	int m = nStr.size();
	vector<vector<int>> dp(m, vector<int>(1 << 10, -1));
	function<int(int, int, bool, bool)> f = [&](int i, int mask, bool islimit, bool isnum)->int {
		if (i == m) {
			return (int)isnum;
		}
		if (!islimit && isnum && dp[i][mask] != -1) {
			return dp[i][mask];
		}
		int res = 0;
		if (!isnum) {
			res = f(i + 1, mask, false, isnum);
		}
		int up = islimit ? nStr[i] - '0' : 9;
		for (int d = 1 - int(isnum); d <= up; d++) {
			if ((mask >> d & 1) == 1) {
				continue;
			}
			res += f(i + 1, mask | (1 << d), islimit && d == nStr[i] - '0', true);
		}
		if (!islimit && isnum) {
			dp[i][mask] = res;
		}
		return res;
		};

	return f(0, 0, true, false);
}

int LClabuladong::countDigitOne(int n)
{
	string nstr = to_string(n);
	int m = nstr.size();
	vector<vector<int>> dp(m, vector<int>(m, -1));
	function<int(int, bool, bool, int)> f = [&](int i, bool limited, bool isnum, int numone)->int {
		if (i == m) {
			return numone;
		}
		if (!limited && isnum && dp[i][numone] != -1) {
			return dp[i][numone];
		}
		int res = 0;
		int up = limited ? nstr[i] - '0' : 9;
		if (!isnum) {
			res += f(i + 1, false, false, numone);
		}
		for (int d = 1 - (int)isnum; d <= up; d++) {
			res += f(i + 1, limited && d == nstr[i] - '0', true, d == 1 ? numone + 1 : numone);
		}
		if (!limited && isnum) {
			dp[i][numone] = res;
		}
		return res;
		};

	return f(0, true, false, 0);
}

int LClabuladong::atMostNGivenDigitSet(vector<string>& digits, int n)
{
	string nstr = to_string(n);
	int m = nstr.size();
	vector<int> memo(m, -1);
	function<int(int, bool, bool)> f = [&](int i, bool limited, bool isnum)->int {
		if (i == m) {
			return (int)isnum;
		}
		if (!limited && isnum && memo[i] != -1) {
			return memo[i];
		}
		int res = 0;
		int index = 0;
		if (limited) {
			while (index < digits.size() && digits[index][0] - '0' <= nstr[i] - '0') {
				index++;
			}
		}
		index = limited ? index : digits.size();
		if (!isnum) {
			res += f(i + 1, false, false);
		}
		for (int d = 0; d < index; d++) {
			res += f(i + 1, limited && digits[d][0] == nstr[i], true);
		}
		if (!limited && isnum) {
			memo[i] = res;
		}
		return res;
		};

	return f(0, true, false);
}

int LClabuladong::findIntegers(int n)
{
	bitset<32> nbit(n);
	int m = nbit.size();
	vector<vector<int>> memo(m, vector<int>(2, -1));
	function<int(int, bool, bool)> f = [&](int i, bool mask, bool limited)->int {
		if (i == m) {
			return 1;
		}
		if (!limited && memo[i][mask] != -1) {
			return memo[i][mask];
		}
		int res = 0;
		bool t = limited ? nbit[31 - i] : true;
		int up = t && mask ? 1 : 0;
		for (int d = 0; d <= up; d++) {
			res += f(i + 1, d == 0, limited && d == nbit[31 - i]);
		}
		if (!limited) {
			memo[i][(int)mask] = res;
		}

		return res;
		};

	return f(0, 0, true);
}

vector<string> LClabuladong::splitWordsBySeparator(vector<string>& words, char separator)
{
	vector<string> res;
	for (string& str : words) {
		int index = str.find(separator, 0);
		int last = 0;
		while (index!=-1) {
			string t = str.substr(last, index - last);
			if (!t.empty()) {
				res.push_back(t);
			}
			last = index + 1;
			index = str.find(separator, last);
		}
		string t = str.substr(last, str.size() - last);
		if (!t.empty()) {
			res.push_back(t);
		}
	}
	return res;
}

long long LClabuladong::maximumSumOfHeights(vector<int>& maxHeights)
{
	int n = maxHeights.size();
	long long res = 0;
	vector<long long> prefix(n), suffix(n);
	stack<int> stack1, stack2;

	for (int i = 0; i < n; i++) {
		while (!stack1.empty() && maxHeights[i] < maxHeights[stack1.top()]) {
			stack1.pop();
		}
		if (stack1.empty()) {
			prefix[i] = (long long)(i + 1) * maxHeights[i];
		}
		else {
			prefix[i] = prefix[stack1.top()] + (long long)(i - stack1.top()) * maxHeights[i];
		}
		stack1.emplace(i);
	}
	for (int i = n - 1; i >= 0; i--) {
		while (!stack2.empty() && maxHeights[i] < maxHeights[stack2.top()]) {
			stack2.pop();
		}
		if (stack2.empty()) {
			suffix[i] = (long long)(n - i) * maxHeights[i];
		}
		else {
			suffix[i] = suffix[stack2.top()] + (long long)(stack2.top() - i) * maxHeights[i];
		}
		stack2.emplace(i);
		res = max(res, prefix[i] + suffix[i] - maxHeights[i]);
	}
	return res;
}

int LClabuladong::rangeSumBST(TreeNode* root, int low, int high)
{
	if (!root) {
		return 0;
	}
	int ret = 0;
	if (root->val<=high && root->val>=low) {
		ret += root->val;
		if(root->val != high)
		ret += rangeSumBST(root->right, low, high);
		if (root->val != low)
		ret += rangeSumBST(root->left, low, high);
	}
	if (root->val < low) {
		ret += rangeSumBST(root->right, low, high);
	}
	if (root->val > high) {
		ret += rangeSumBST(root->left, low, high);
	}

	return ret;
}

long long LClabuladong::countPaths(int n, vector<vector<int>>& edges)
{
	countPaths_isPrime = new bool[countPaths_MAX + 1];
	for (int i = 0; i < countPaths_MAX + 1;i++) {
		countPaths_isPrime[i] = true;
	}
	countPaths_isPrime[1] = false;
	for (int i = 1; i * i < countPaths_MAX; i++) {
		if (countPaths_isPrime[i]) {
			for (int j = i * i; j <= countPaths_MAX; j += i) {
				countPaths_isPrime[j] = false;
			}
		}
	}
	vector<vector<int>> g(n+1);
	for (auto& e : edges) {
		g[e[0]].push_back(e[1]);
		g[e[1]].push_back(e[0]);
	}
	vector<int> size1(n + 1, 0);
	vector<int> nodes;
	function<void(int, int)> dfs = [&](int x, int fa)->void {
		nodes.push_back(x);
		for (int y : g[x]) {
			if (y != fa && !countPaths_isPrime[y]) {
				dfs(y, x);
			}
		}
		};
	long long ret = 0;
	for (int i = 1; i <= n; i++) {
		if (!countPaths_isPrime[i]) {
			continue;
		}
		long long sum = 0;
		for (int& j: g[i]) {
			if (countPaths_isPrime[j]) {
				continue;
			}
			if (size1[j] == 0) {
				nodes.clear();
				dfs(j, -1);
			}
			for (int& z : nodes) {
				size1[z] = nodes.size();
			}
		}
		for (int& j : g[i]) {
			ret += (sum * size1[j]);
			sum += size1[j];
		}
		ret += sum;
	}

	return ret;
}

int LClabuladong::minIncrements(int n, vector<int>& cost)
{
	int ret = 0;
	function<int(int)> recursion = [&](int root)->int {
		if (root * 2 > n) {
			return cost[root - 1];
		}
		int leftcost = recursion(root * 2) + cost[root - 1];
		int rightcost = recursion(root * 2 + 1) + cost[root - 1];
		int diff = abs(leftcost - rightcost);
		ret += diff;
		return max(leftcost,rightcost);
		};
	recursion(1);

	return ret;
}

int LClabuladong::distinctIntegers(int n)
{
	if (n <= 1) return 1;
	return n - 1;
}

vector<int> LClabuladong::sumOfDistancesInTree(int n, vector<vector<int>>& edges)
{
	vector<int> sz(n), dp(n), ans(n);
	vector<vector<int>> graph(n, vector<int>());
	for (auto& e : edges) {
		graph[e[0]].push_back(e[1]);
		graph[e[1]].push_back(e[0]);
	}
	function<void(int, int)> dfs = [&](int u, int f)->void {
		int ret = 0;
		int nodesize = 1;
		for (auto& c : graph[u]) {
			if (c == f) {
				continue;
			}
			dfs(c, u);
			nodesize += sz[c];
			ret += dp[c];
		}
		ret = ret + nodesize - 1;
		sz[u] = nodesize;
		dp[u] = ret;
		};
	function<void(int, int)> dfs2 = [&](int u, int f)->void {
		ans[u] = dp[u];
		for (auto& v : graph[u]) {
			if (v == f) {
				continue;
			}
			//保留原来根信息
			int pu = dp[u], pv = dp[v];
			int su = sz[u], sv = sz[v];

			//换根算法
			dp[u] -= dp[v] + sz[v];
			sz[u] -= sz[v];
			dp[v] += dp[u] + sz[u];
			sz[v] += sz[u];

			dfs2(v, u);

			//复原根
			dp[u] = pu, dp[v] = pv;
			sz[u] = su, sz[v] = sv;
		}
		};

	dfs(0, -1);
	dfs2(0, -1);

	return ans;
}

bool LClabuladong::validPartition(vector<int>& nums)
{
	vector<bool> isvalid(nums.size(), false);
	if (nums[0] == nums[1]) {
		isvalid[1] = true;
	}
	if (nums.size() == 2) {
		return isvalid[1];
	}
	if ((nums[0] == nums[1] && nums[0] == nums[2]) || (nums[0] + 1 == nums[1] && nums[0] + 2 == nums[2])) {
		isvalid[2] = true;
	}
	for (int i = 3; i < isvalid.size(); i++) {
		isvalid[i] = (isvalid[i - 2] && nums[i] == nums[i - 1]);
		if (isvalid[i - 3]) {
			if ((nums[i - 2] == nums[i - 1] && nums[i - 2] == nums[i]) || (nums[i - 2] + 1 == nums[i - 1] && nums[i - 2] + 2 == nums[i])) {
				isvalid[i] = true;
			}
		}
	}

	return isvalid[isvalid.size() - 1];
}

int LClabuladong::reachableNodes(int n, vector<vector<int>>& edges, vector<int>& restricted)
{
	vector<vector<int>> graph(n, vector<int>());
	unordered_set<int> rs;
	for (auto& node : restricted) {
		rs.insert(node);
	}
	for (auto& e : edges) {
		if (!(rs.count(e[0]) || rs.count(e[1]))) {
			graph[e[0]].push_back(e[1]);
			graph[e[1]].push_back(e[0]);
		}
	}
	function<int(int, int)> getNode = [&](int root, int fa)->int {
		int ret = 1;
		for(auto& node:graph[root]){
			if (node == fa) {
				continue;
			}
			ret += getNode(node, root);
		}
		return ret;
		};

	return getNode(0, -1);
}

int LClabuladong::minimizeArrayValue(vector<int>& nums)
{
	int l = nums[0], r = 1e9;
	vector<long long> tem(nums.size());
	for (int i = 0; i < tem.size(); i++) {
		tem[i] = nums[i];
	}
	while (l < r) {
		int mid = (r - l) / 2 + l;
		vector<long long> t = tem;
		for (int i = nums.size() - 1 ; i > 0; i--) {
			if (t[i] <= mid) {
				continue;
			}
			t[i - 1] += (t[i] - mid);
		}
		if (t[0] <= mid) {
			r = mid;
		}
		else {
			l = mid + 1;
		}
	}

	return r;
}

int LClabuladong::countThePaths(int n, vector<vector<int>>& roads)
{
	const long long mod = 1e9 + 7;
	vector<vector<pair<int, int>>> e(n);
	for (const auto& road : roads) {
		int x = road[0], y = road[1], t = road[2];
		e[x].emplace_back(y, t);
		e[y].emplace_back(x, t);
	}

	vector<long long> dis(n, LLONG_MAX);
	vector<long long> ways(n);

	priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> q;
	q.emplace(0, 0);
	dis[0] = 0;
	ways[0] = 1;
	
	while (!q.empty()) {
		pair<long long, int> l = q.top();
		q.pop();
		for (pair<int, int>& p : e[l.second]) {
			if (dis[p.first] < l.first + p.second) {
				continue;
			}
			if (dis[p.first] > l.first + p.second) {
				dis[p.first] = l.first + p.second;
				ways[p.first] = ways[l.second];
				q.emplace(dis[p.first], p.first);
			}else 
			if (dis[p.first] == l.first + p.second) {
				ways[p.first] = (ways[l.second] + ways[p.first]) % mod;
			}
		}
	}

	return ways[n-1];
}

int LClabuladong::findKOr(vector<int>& nums, int k)
{
	int ret = 0;
	for (int i = 0; i < 32; i++) {
		int t = 0;
		for (auto a : nums) {
			if ((a >> i & 1) == 1) {
				t++;
			}
		}
		if (t >= k) {
			ret += (1 << i);
		}
	}

	return ret;
}

vector<int> LClabuladong::divisibilityArray(string word, int m)
{
	int n = word.size();
	vector<int> ret(n, 0);
	long long total = 0;
	for (int i = 0; i < n; i++) {
		total = (total * 10 + (word[i] - '0')) % m;
		if (total == 0) {
			ret[i] = 1;
		}
	}

	return ret;
}


