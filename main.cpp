#include "LClabuladong.h"

int main()
{
	LClabuladong solution;
	vector<string> d1_arr;
	vector<int> d2_str;
	vector<int> d1_arr2 = { 7,12,9,8,9,15 };
	vector<string> d1_arr3 = { "one.two.three","four.five","six" };
	vector<vector<int>> d2_arr = {{0, 6, 7}, {0, 1, 2}, {1, 2, 3}, {1, 3, 3}, {6, 3, 3}, {3, 5, 1}, {6, 5, 1}, {2, 5, 1}, {0, 4, 5}, {4, 6, 2}};
	string str1 = "ABFCACDB", str2 = "2";
	ListNode* root = new ListNode(d1_arr2);
	auto ans = solution.findKOr(d1_arr2, 4);

	return 0;
}
