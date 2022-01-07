#ifndef KDTREE_H
#define KDTREE_H

#include "hitableArray.h"
#include "utils/stack.h"

namespace cuda::rayTracing {

template <typename iterator>
__host__ __device__ box calcBox(iterator begin, iterator end){
    box resbox;
    for(auto it = begin; it != end; it++){
        const box itbox = (*it)->getBox();
        resbox.min = min(itbox.min, resbox.min);
        resbox.max = max(itbox.max, resbox.max);
    }
    return resbox;
}

template <typename iterator>
struct KDNode{
    KDNode* left{nullptr};
    KDNode* right{nullptr};
    iterator begin;
    size_t size{0};
    box bbox;

    static constexpr size_t itemsInNode = 10;
    static constexpr size_t stackSize = 50;

    __host__ __device__ KDNode() {}
    __host__ __device__ ~KDNode() {
        if(left) delete left;
        if(right) delete right;
    }
    __host__ __device__ KDNode(const KDNode& other) = delete;
    __host__ __device__ KDNode& operator=(const KDNode& other) = delete;

    __host__ __device__ KDNode(KDNode&& other) : left(other.left), right(other.right), begin(other.begin), size(other.size), bbox(other.bbox) {
        other.left = nullptr; other.right = nullptr; other.size = 0;
    }
    __host__ __device__ KDNode& operator=(KDNode&& other)
    {
        left = other.left; right = other.right; begin = other.begin; size = other.size; bbox = other.bbox;
        other.left = nullptr; other.right = nullptr; other.size = 0;
        return *this;
    }

    __host__ __device__ iterator end() const { return begin + size;}
    __host__ __device__ bool check() const { return size > itemsInNode;}

    __host__ KDNode(iterator begin, size_t size) : begin(begin), size(size), bbox(calcBox(begin, begin + size)){
        if(const iterator end = begin + size; size > itemsInNode)
        {
            sortByBox(begin, end, bbox);

            float bestSAH = std::numeric_limits<float>::max();
            size_t bestSize = 0, itSize = 0;
            for(auto curr = begin; curr != end; curr++){
                size_t leftN = ++itSize;
                size_t rightN = size - itSize;

                float leftS = calcBox(begin, curr + 1).surfaceArea();
                float rightS = calcBox(curr + 1, end).surfaceArea();

                if(float SAH = leftN * leftS + rightN * rightS; SAH < bestSAH){
                    bestSAH = SAH;
                    bestSize = itSize;
                }
            }

            left = new KDNode(begin, bestSize);
            right = new KDNode(begin + bestSize, size - bestSize);
        }
    }

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) {
        Stack<const KDNode*, KDNode::stackSize> selected;
        for(Stack<const KDNode*, KDNode::stackSize> traversed(this); !traversed.empty();){
            if(const KDNode* curr = traversed.top(); traversed.pop() && curr->bbox.intersect(r)){
                if (!curr->left && !curr->right){
                    selected.push(curr);
                } else {
                    traversed.push(curr->left);
                    traversed.push(curr->right);
                }
            }
        }

        for(auto curr = selected.top(); selected.pop(); curr = selected.top()){
            for(iterator it = curr->begin; it != curr->end(); it++){
                if ((*it)->hit(r, coord)) coord.obj = *it;
            }
        }

        return coord.obj;
    }
};

template <typename iterator>
size_t findMaxDepth(KDNode<iterator>* node, size_t& index){
    if(node){
        size_t res = ++index;
        res = std::max(res, findMaxDepth(node->left, index));
        res = std::max(res, findMaxDepth(node->right, index));
        --index;
        return res;
    }
    return 0;
}

struct NodeDescriptor{
    uint32_t size{0};
    box bbox{};
    uint32_t offset{0};
    uint32_t curr{0};
    uint32_t left{0};
    uint32_t right{0};
};

template <typename iterator>
void buildNodeDescriptorsRecursive(KDNode<iterator>* node, std::vector<NodeDescriptor>& nodeDescriptors, uint32_t& counter, uint32_t& offset, uint32_t curr){
    if(node){
        NodeDescriptor nodeDescriptor{};
        nodeDescriptor.curr = curr;
        nodeDescriptor.size = node->size;
        nodeDescriptor.bbox = node->bbox;
        nodeDescriptor.offset = offset;
        nodeDescriptor.right = node->check() ? counter++ : counter;
        nodeDescriptor.left = node->check() ? counter++ : counter;
        nodeDescriptors.push_back(nodeDescriptor);

        offset += node->check() ? 0 : node->size;
        buildNodeDescriptorsRecursive(node->left, nodeDescriptors, counter, offset, nodeDescriptor.left);
        buildNodeDescriptorsRecursive(node->right, nodeDescriptors, counter, offset, nodeDescriptor.right);
    }
}

template <typename container>
class KDTree{
private:
    KDNode<typename container::iterator>* root{nullptr};
    size_t maxDepth{0};

public:
    container storage;

    __host__ KDTree() {}
    __host__ ~KDTree(){
        if(root) delete root;
    }
    __host__ void makeTree(){
        root = new KDNode<typename container::iterator>(storage.begin(), storage.size());
        maxDepth = findMaxDepth(root, maxDepth);
    }
    KDNode<typename container::iterator>* getRoot() const {
        return root;
    }

    std::vector<NodeDescriptor> buildNodeDescriptors() const {
        std::vector<NodeDescriptor> nodeDescriptors;
        uint32_t offset = 0;
        uint32_t counter = 1;
        buildNodeDescriptorsRecursive(root, nodeDescriptors, counter, offset, 0);
        return nodeDescriptors;
    }
};

class HitableKDTree : public HitableContainer {
public:
    using container = HitableArray;

private:
    container* storage{nullptr};
    KDNode<container::iterator>* root{nullptr};

public:
    using iterator = container::iterator;
    using KDNodeType = KDNode<container::iterator>;

    __host__ __device__ HitableKDTree() {
        storage = new container();
    };
    __host__ __device__ ~HitableKDTree(){
        if(storage) delete storage;
        if(root)    delete [] root;
    }

    __host__ __device__ void setRoot(KDNodeType* root){
        this->root = root;
    }

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) const override{
        return root->hit(r, coord);
    }

    __host__ __device__ void add(const Hitable*const* objects, size_t size = 1) override{
        storage->add(objects, size);
    }

    __host__ __device__ const Hitable*& operator[](uint32_t i) const override{
        return (*storage)[i];
    }

    __host__ __device__ void makeTree(KDNodeType* nodesBuffer, NodeDescriptor nodeDescriptor) {
        KDNodeType* curr = &nodesBuffer[nodeDescriptor.curr];

        curr->begin = storage->begin() + nodeDescriptor.offset;
        curr->size = nodeDescriptor.size;
        curr->bbox = nodeDescriptor.bbox;
        curr->right = curr->check() ? &nodesBuffer[nodeDescriptor.right] : nullptr;
        curr->left = curr->check() ? &nodesBuffer[nodeDescriptor.left] : nullptr;
    }

    __host__ __device__ iterator begin() {return storage->begin(); }
    __host__ __device__ iterator end() {return storage->end(); }

    __host__ __device__ iterator begin() const {return storage->begin(); }
    __host__ __device__ iterator end() const {return storage->end(); }

    static void create(HitableKDTree* dpointer, const HitableKDTree& host);
    static void destroy(HitableKDTree* dpointer);
};

void makeTree(HitableKDTree* container, NodeDescriptor* nodeDescriptors, size_t size);

}
#endif // KDTREE_H
