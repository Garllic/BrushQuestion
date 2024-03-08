#include "tool.h"

ListNode::ListNode(std::vector<int>& nums)
{
	this->val = nums[0];
	ListNode* p = this;
	for (int i = 1; i < nums.size(); i++) {
		ListNode* temp = new ListNode(nums[i]);
		p->next = temp;
		p = p->next;
	}
}

TreeNode::TreeNode(std::vector<int> arr)
{
	std::queue<TreeNode*> q;
	TreeNode* p;
	TreeNode* temp;
	this->val = arr[0];
	q.push(this);
	for (int i = 1; i < arr.size(); i++) {
		p = q.front();
		q.pop();
		if (p == NULL) {
			--i;
			continue;
		}

		if (arr[i] != INT_MAX) {
			temp = new TreeNode();
			temp->val = arr[i++];
			p->left = temp;
			q.push(p->left);
		}
		else {
			q.push(NULL);
			++i;
		}

		if (i == arr.size()) {
			break;
		}

		if (arr[i] != INT_MAX) {
			temp = new TreeNode();
			temp->val = arr[i];
			p->right = temp;
			q.push(p->right);
		}
		else {
			q.push(NULL);
		}

	}
}

std::vector<int> TreeNode::ToVector()
{
	std::vector<int> res;
	std::queue<TreeNode*> myq;
	TreeNode* temp;
	res.push_back(this->val);
	myq.push(this->left);
	myq.push(this->right);
	int flag = 2;
	while (flag) {
		temp = myq.front();
		myq.pop();

		if (temp == NULL) {
			res.push_back(INT_MAX);
			continue;
		}
		--flag;

		res.push_back(temp->val);
		myq.push(temp->left);
		if (temp->left != NULL) {
			++flag;
		}
		myq.push(temp->right);
		if (temp->right != NULL) {
			++flag;
		}
	}
	return res;
}

Node::Node(int _val) {
	val = _val;
}

Node::Node(int _val, std::vector<Node*> _children) {
	val = _val;
	children = _children;
}

UnionFind::UnionFind(int n)
{
	parent.resize(n);
	size.resize(n, 1);
	nUnion = n;
	for (int i = 0; i < n; i++) {
		parent[i] = i;
	}

}

void UnionFind::connect(int p, int q)
{
	int rootP = findRoot(p);
	int rootQ = findRoot(q);
	if (isConnected(rootP, rootQ)) {
		return;
	}
	if (size[rootP] > size[rootQ]) {
		parent[rootQ] = rootP;
		size[rootP] += size[rootQ];
	}
	else {
		parent[rootP] = rootQ;
		size[rootQ] = rootP;
	}
	nUnion--;
}

bool UnionFind::isConnected(int p, int q)
{
	int rootP = findRoot(p);
	int rootQ = findRoot(q);

	return rootP == rootQ;
}

int UnionFind::findRoot(int x)
{
	int root = x;
	while (root != parent[root]) {
		root = parent[x];
	}

	return root;
}

int UnionFind::GetUnionNum()
{
	return nUnion;
}
