#ifndef SCENE_H
#define SCENE_H

#include <cstdint>

class scene
{
public:
    virtual ~scene(){};

    virtual void resize() = 0;
    virtual void updateFrame(uint32_t frameNumber, float frameTime) = 0;
};

#endif // SCENE_H
