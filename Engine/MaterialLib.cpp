#include <StdAfx.h>
#include "MaterialLib.h"

#define S(X) (X / 100.0f)

namespace CudaTracerLib {

static MaterialEntry materialData[] = {
	/* Fitted data from "A Practical Model for Subsurface scattering" (Jensen et al.). No anisotropy data available. */
	{ "Apple",                      { S(2.29f), S(2.39f), S(1.97f) }, { S(0.0030f), S(0.0034f), S(0.0460f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Chicken1",                   { S(0.15f), S(0.21f), S(0.38f) }, { S(0.0015f), S(0.0770f), S(0.1900f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Chicken2",                   { S(0.19f), S(0.25f), S(0.32f) }, { S(0.0018f), S(0.0880f), S(0.2000f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Cream",                      { S(7.38f), S(5.47f), S(3.15f) }, { S(0.0002f), S(0.0028f), S(0.0163f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Ketchup",                    { S(0.18f), S(0.07f), S(0.03f) }, { S(0.0610f), S(0.9700f), S(1.4500f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Marble",                     { S(2.19f), S(2.62f), S(3.00f) }, { S(0.0021f), S(0.0041f), S(0.0071f) }, { 0.0f, 0.0f, 0.0f }, 1.5f },
	{ "Potato",                     { S(0.68f), S(0.70f), S(0.55f) }, { S(0.0024f), S(0.0090f), S(0.1200f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Skimmilk",                   { S(0.70f), S(1.22f), S(1.90f) }, { S(0.0014f), S(0.0025f), S(0.0142f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Skin1",                      { S(0.74f), S(0.88f), S(1.01f) }, { S(0.0320f), S(0.1700f), S(0.4800f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Skin2",                      { S(1.09f), S(1.59f), S(1.79f) }, { S(0.0130f), S(0.0700f), S(0.1450f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Spectralon",                 { S(11.6f), S(20.4f), S(14.9f) }, { S(0.0000f), S(0.0000f), S(0.0000f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },
	{ "Wholemilk",                  { S(2.55f), S(3.21f), S(3.77f) }, { S(0.0011f), S(0.0024f), S(0.0140f) }, { 0.0f, 0.0f, 0.0f }, 1.3f },

	/* From "Acquiring Scattering Properties of Participating Media by Dilution"
	   by Narasimhan, Gupta, Donner, Ramamoorthi, Nayar, Jensen (SIGGRAPH 2006) */
	{ "Lowfat Milk",				{ 0.9124f, 1.0744f, 1.2492f }, { 0.9126f, 1.0748f, 1.2500f }, { 0.9320f, 0.9020f, 0.8590f }, 1.33f },
	{ "Reduced Milk",				{ 1.0748f, 1.2209f, 1.3931f }, { 1.0750f, 1.2213f, 1.3941f }, { 0.8190f, 0.7970f, 0.7460f }, 1.33f },
	{ "Regular Milk",				{ 1.1873f, 1.3293f, 1.4589f }, { 1.1874f, 1.3296f, 1.4602f }, { 0.7500f, 0.7140f, 0.6810f }, 1.33f },
	{ "Espresso",					{ 0.2707f, 0.2828f, 0.2970f }, { 0.4376f, 0.5115f, 0.6048f }, { 0.9070f, 0.8960f, 0.8800f }, 1.33f },
	{ "Mint Mocha Coffee",			{ 0.0916f, 0.1081f, 0.1460f }, { 0.1900f, 0.2600f, 0.3500f }, { 0.9100f, 0.9070f, 0.9140f }, 1.33f },
	{ "Lowfat Soy Milk",			{ 0.1418f, 0.1620f, 0.2715f }, { 0.1419f, 0.1625f, 0.2740f }, { 0.8500f, 0.8530f, 0.8420f }, 1.33f },
	{ "Regular Soy Milk",			{ 0.2433f, 0.2714f, 0.4563f }, { 0.2434f, 0.2719f, 0.4597f }, { 0.8730f, 0.8580f, 0.8320f }, 1.33f },
	{ "Lowfat Chocolate Milk",		{ 0.4277f, 0.4998f, 0.5723f }, { 0.4282f, 0.5014f, 0.5791f }, { 0.9340f, 0.9270f, 0.9160f }, 1.33f },
	{ "Regular Chocolate Milk",		{ 0.7352f, 0.9142f, 1.0588f }, { 0.7359f, 0.9172f, 1.0688f }, { 0.8620f, 0.8380f, 0.8060f }, 1.33f },
	{ "Coke",						{ 0.0177f, 0.0208f, 0.0000f }, { 0.7143f, 1.1688f, 1.7169f }, { 0.9650f, 0.9720f, 0.9685f }, 1.33f },
	{ "Pepsi",						{ 0.0058f, 0.0141f, 0.0000f }, { 0.6433f, 0.9990f, 1.4420f }, { 0.9260f, 0.9790f, 0.9525f }, 1.33f },
	{ "Sprite",						{ 0.0069f, 0.0089f, 0.0089f }, { 0.1299f, 0.1283f, 0.1395f }, { 0.9430f, 0.9530f, 0.9520f }, 1.33f },
	{ "Gatorade",					{ 0.2392f, 0.2927f, 0.3745f }, { 0.4009f, 0.4185f, 0.4324f }, { 0.9330f, 0.9330f, 0.9350f }, 1.33f },
	{ "Chardonnay",					{ 0.0030f, 0.0047f, 0.0069f }, { 0.1577f, 0.1748f, 0.3512f }, { 0.9140f, 0.9580f, 0.9750f }, 1.33f },
	{ "White Zinfandel",			{ 0.0031f, 0.0048f, 0.0066f }, { 0.1763f, 0.2370f, 0.2913f }, { 0.9190f, 0.9430f, 0.9720f }, 1.33f },
	{ "Merlot",						{ 0.0053f, 0.0000f, 0.0000f }, { 1.6429f, 1.9196f, 0.0053f }, { 0.9740f, 0.9740f, 0.9740f }, 1.33f },
	{ "Budweiser Beer",				{ 0.0037f, 0.0069f, 0.0074f }, { 0.1486f, 0.3210f, 0.7360f }, { 0.9170f, 0.9560f, 0.9820f }, 1.33f },
	{ "Coors Light Beer",			{ 0.0027f, 0.0055f, 0.0000f }, { 0.0295f, 0.0663f, 0.1521f }, { 0.9180f, 0.9660f, 0.9420f }, 1.33f },
	{ "Clorox",						{ 0.1425f, 0.1723f, 0.1928f }, { 0.1600f, 0.2500f, 0.3300f }, { 0.9120f, 0.9050f, 0.8920f }, 1.33f },
	{ "Apple Juice",				{ 0.0201f, 0.0243f, 0.0323f }, { 0.1215f, 0.2101f, 0.4407f }, { 0.9470f, 0.9490f, 0.9450f }, 1.33f },
	{ "Cranberry Juice",			{ 0.0128f, 0.0155f, 0.0196f }, { 0.2700f, 0.6300f, 0.8300f }, { 0.9470f, 0.9510f, 0.9740f }, 1.33f },
	{ "Grape Juice",				{ 0.0072f, 0.0000f, 0.0000f }, { 1.2500f, 1.5300f, 0.0072f }, { 0.9610f, 0.9610f, 0.9610f }, 1.33f },
	{ "Ruby Grapefruit Juice",		{ 0.1617f, 0.1606f, 0.1669f }, { 0.2513f, 0.3517f, 0.4305f }, { 0.9290f, 0.9290f, 0.9310f }, 1.33f },
	{ "White Grapefruit Juice",		{ 0.3513f, 0.3669f, 0.5237f }, { 0.3609f, 0.3800f, 0.5632f }, { 0.5480f, 0.5450f, 0.5650f }, 1.33f },
	{ "Shampoo",					{ 0.0104f, 0.0114f, 0.0147f }, { 0.0288f, 0.0710f, 0.0952f }, { 0.9100f, 0.9050f, 0.9200f }, 1.33f },
	{ "Strawberry Shampoo",			{ 0.0028f, 0.0032f, 0.0033f }, { 0.0217f, 0.0788f, 0.1022f }, { 0.9270f, 0.9350f, 0.9940f }, 1.33f },
	{ "Head & Shoulders Shampoo",	{ 0.2791f, 0.2890f, 0.3086f }, { 0.3674f, 0.4527f, 0.5211f }, { 0.9110f, 0.8960f, 0.8840f }, 1.33f },
	{ "Lemon Tea Powder",			{ 0.0798f, 0.0898f, 0.1073f }, { 0.3400f, 0.5800f, 0.8800f }, { 0.9460f, 0.9460f, 0.9490f }, 1.33f },
	{ "Orange Juice Powder",		{ 0.1928f, 0.2132f, 0.2259f }, { 0.3377f, 0.5573f, 1.0122f }, { 0.9190f, 0.9180f, 0.9220f }, 1.33f },
	{ "Pink Lemonade Powder",		{ 0.1235f, 0.1334f, 0.1305f }, { 0.2400f, 0.3700f, 0.4500f }, { 0.9020f, 0.9020f, 0.9040f }, 1.33f },
	{ "Cappuccino Powder",			{ 0.0654f, 0.0882f, 0.1568f }, { 0.2574f, 0.3536f, 0.4840f }, { 0.8490f, 0.8430f, 0.9260f }, 1.33f },
	{ "Salt Powder",				{ 0.2485f, 0.2822f, 0.3216f }, { 0.7600f, 0.8685f, 0.9363f }, { 0.8020f, 0.7930f, 0.8210f }, 1.33f },
	{ "Sugar Powder",				{ 0.0145f, 0.0162f, 0.0202f }, { 0.0795f, 0.1759f, 0.2780f }, { 0.9210f, 0.9190f, 0.9310f }, 1.33f },
	{ "Suisse Mocha",				{ 0.3223f, 0.3583f, 0.4148f }, { 0.5098f, 0.6476f, 0.7944f }, { 0.9070f, 0.8940f, 0.8880f }, 1.33f },
};

MaterialEntry* MaterialLibrary::getMat(const std::string& name)
{
	MaterialEntry *matEntry = materialData;
	for (int i = 0; i < sizeof(materialData) / sizeof(MaterialEntry); i++)
		if (materialData[i].name == name)
			return materialData + i;
	return 0;
}

size_t MaterialLibrary::getNumMats()
{
	return sizeof(materialData) / sizeof(materialData[0]);
}

const std::string& MaterialLibrary::getMatName(size_t idx)
{
	return materialData[idx].name;
}

}