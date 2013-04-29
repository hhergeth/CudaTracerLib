#pragma once

#include "Base\FileStream.h"
#include "Engine\e_DynamicScene.h"
#include "Engine\e_Camera.h"

class SkyrimImporter
{
public:
	static void ReadEsm(e_DynamicScene* S, e_Camera* C);
};