#pragma once
#include <vector>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <string>
#include <stack>
#include <math.h>
#include <cmath>
#include <map>
#include <queue>
#include <Windows.h>
#include <unordered_set>
#include <ctime>
#include <functional>
#include <set>
#include <bitset>

//辅助类
/*---------------------------------------------------------------------------------*/

//链表
class ListNode {
public:
	int val;
	ListNode* next;

	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}
	ListNode(std::vector<int>& nums);
};

//树
class TreeNode
{
public:
	int val;
	TreeNode* left;
	TreeNode* right;

	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
	TreeNode(std::vector<int> arr);

	std::vector<int> ToVector();
};

class Node
{
public:
	int val;
	std::vector<Node*> children;

	Node(int _val);
	Node(int _val, std::vector<Node*> _children);
};

class Node2 {
public:
	int val;
	Node2* left;
	Node2* right;
	Node2* next;

	Node2() : val(0), left(NULL), right(NULL), next(NULL) {}

	Node2(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

	Node2(int _val, Node2* _left, Node2* _right, Node2* _next)
		: val(_val), left(_left), right(_right), next(_next) {}
};

//并查集合
class UnionFind
{
private:
    int nUnion;
    std::vector<int> parent;
    std::vector<int> size;
public:
    UnionFind(int n);
    //将p q连通
    void connect(int p, int q);
    //判断p  q是否连通
	bool isConnected(int p, int q);
    //寻找x的根节点
	int findRoot(int x);
    //返回连通分量个数
	int GetUnionNum();
};

//辅助函数
/*---------------------------------------------------------------------------------*/

//打印二维向量
template<typename T>
void printMatrix(std::vector<std::vector<T>>& m)
{
	int row = m.size();

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < m[i].size(); j++) {
			std::cout << std::left << std::setw(5) << m[i][j];
		}
		std::cout << std::endl;
	}
}

template<typename T>
void printVector(std::vector<T>& v)
{
	for (T& t : v) {
		std::cout << t << ' ';
	}
	std::cout << std::endl;
}
