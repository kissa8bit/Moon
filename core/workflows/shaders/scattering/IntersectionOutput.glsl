#ifndef SCATTERING_INTERSECTION_OUTPUT
#define SCATTERING_INTERSECTION_OUTPUT

struct IntersectionOutput{
    bool intersectionCondition;
    bool inside;
    float intersectionPoint1;
    float intersectionPoint2;
};

IntersectionOutput IntersectionOutputDef() {
    IntersectionOutput res;
    res.intersectionCondition = false;
    res.inside = false;
    res.intersectionPoint1 = 0.0f;
    res.intersectionPoint2 = 0.0f;
    return res;
}

#endif