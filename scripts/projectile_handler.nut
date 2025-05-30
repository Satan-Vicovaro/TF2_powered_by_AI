// Globals
::FLT_MAX <- 3.402823466e+38;
::distances <- {};
::min_positions <- {};
::min_positions_diffs <- {};
::TRACKING_RATE <- 0.001

const debug = true;

IncludeScript("file_scripts");

local data_modes = {
    POS_ABS = "pos_abs",
    DISTANCE = "distance",
    POS_DIFF = "pos_diff"
};

local information_mode = data_modes.DISTANCE;

function ClearStringFromPool(string)
{
	local dummy = Entities.CreateByClassname("info_target")
	dummy.KeyValueFromString("targetname", string)
	NetProps.SetPropBool(dummy, "m_bForcePurgeFixedupStrings", true)
	dummy.Destroy()
}

function EntFireCodeSafe(entity, code, delay = 0.0, activator = null, caller = null)
{
	EntFireByHandle(entity, "RunScriptCode", code, delay, activator, caller)
	ClearStringFromPool(code)
}


/*
bool AcceptInput(string input, string param, handle activator, handle caller)
*/
function FireCodeTest(input, param, activator, caller)
{
    return AcceptInput(input, param, activator, caller);
}

function start_tracking(bot_type)
{
    ::distances <- {};
    ::min_positions <- {};
    ::min_positions_diffs <- {};

    local ent = null;
    local projectile_class = null;

    if (bot_type == "soldier") {
        projectile_class = "tf_projectile_rocket";
    } else if (bot_type == "demoman") {
        projectile_class = "tf_projectile_pipe";
    } else {
        printl("[start_tracking] Unknown bot type: " + bot_type);
        return;
    }

    local pos = null;
    local target_bot_position = null;
    local pos_diff = null;
    local target_ent = null;

    while (ent = Entities.FindByClassname(ent, projectile_class))
    {
        if (ent != null)
        {
            local owner = ent.GetOwner();
            if (owner != null)
            {
                EntFireCodeSafe(ent, "track_projectile()", 0.1);
            }
        }
    }
}

function closest_point_on_bbox(pos, ent)
{
    local origin = ent.GetLocalOrigin();
    local mins = ent.GetBoundingMins();
    local maxs = ent.GetBoundingMaxs();

    local clamped = Vector(
        clamp(pos.x, origin.x + mins.x, origin.x + maxs.x),
        clamp(pos.y, origin.y + mins.y, origin.y + maxs.y),
        clamp(pos.z, origin.z + mins.z, origin.z + maxs.z)
    );

    return clamped;
}

function clamp(val, min, max)
{
    return (val < min) ? min : (val > max) ? max : val;
}

function calculate_distance(pos_A, pos_B)
{
    local dx = pos_A.x - pos_B.x;
    local dy = pos_A.y - pos_B.y;
    local dz = pos_A.z - pos_B.z;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

function get_target_bot_entity()
{
    local ent = null;
    while (ent = Entities.FindByClassname(ent, "player"))
    {
        if (ent == GetListenServerHost()) continue;
        if (ent.GetTeam() == TF_TEAM_BLUE) {
            return ent;
        }
    }
    return null;
}

function print_map(map)
{
   foreach(k,v in map)
   {
        printl(k + " : " + v);
   }
}

function track_projectile()
{
    if(debug) printl("tracking projectile")
    local projectile = self; // 'self' is the entity calling this function

    if (projectile == null)
    {
        printl("Error: projectile (self) is null");
        return;
    }

    try {
        if(projectile == null) {printl("Projectile null")}
        local pos = null;
        local owner = null;
        local target_ent = null;

        if(debug) printl("inside try")
        try {
            pos = projectile.GetOrigin();
        } catch(e) {
            printl("pos error " + e);
        }

        try {
            owner = projectile.GetOwner();
        } catch(e) {
            printl("owner error1 " + e);
        }

        try {
            target_ent = get_target_bot_entity();
        } catch(e) {
            printl("target error1 " + e);
        }

        if(debug) printl("afetr tries")

        if(target_ent == null || typeof target_ent != "instance")
        {
            printl("target error2 ");
        }

        local target_bot_position = closest_point_on_bbox(pos, target_ent);
        if(debug) printl("got target's position")

        if (owner != null)
        {
            local ownerIndex = owner.entindex();
            local prevDistance = null;
            if(ownerIndex in ::distances)
                prevDistance = ::distances[ownerIndex];
            local currDistance = calculate_distance(pos, target_bot_position);
            local pos_diff = target_bot_position - pos;

            if(debug) printl("all data gathered")

            if (prevDistance == null || currDistance <= prevDistance)
            {
                ::distances[ownerIndex] <- currDistance;
                ::min_positions[ownerIndex] <- pos;
                ::min_positions_diffs[ownerIndex] <- pos_diff;
                if(debug) printl("found new minimum")
            }
            else {
                // Finishing calculations if the distance doesn't decrease anymore
                return;
            }

            // Debug output
            if(debug) printl("Projectile at " + pos + " |  Target at " + target_bot_position + " | CurrDist: " + currDistance + " | MinDist: " + ::distances[ownerIndex]);

            // Continue updating
            EntFireCodeSafe(projectile, "track_projectile()", TRACKING_RATE);
        }
        else
        {
            printl("owner error2")
        }

    } catch (e) {
        if(debug) printl("Error in track_projectile: " + e);
    }
}

function vector_to_string(vec)
{
    return format("%.3f %.3f %.3f", vec.x, vec.y, vec.z);
}

function send_projectile_info()
{
    if(debug) printl("sending info")

    local projectile_info = "";
    local source_map = null;

    switch(information_mode)
    {
        case data_modes.POS_ABS:
            source_map = ::min_positions
            break;
        case data_modes.POS_DIFF:
            source_map = ::min_positions_diffs
            break;
        case data_modes.DISTANCE:
            source_map = ::distances
            break;
        default :
            printl("wrong information mode")
            return
    }

    if(source_map.len() == 0)
    {
        projectile_info = "b none";
    }
    else
    {
        if(debug) printl("send info else")

        local data_type = null
        foreach(key, val in source_map)
        {
            data_type = typeof val;
            if(debug) printl("data_type: " + data_type)
            break;
        }

        if(data_type == "Vector")
        {

            foreach(ownerIndex, data in source_map)
            {
                if(debug) printl("appending line");
                {
                    projectile_info += format("b %d %s\n", ownerIndex, vector_to_string(data));
                    printl("type of double: " + typeof data);
                }
            }
        }
        else if(data_type == "float")
        {

            foreach(ownerIndex, data in source_map)
            {
                if(debug) printl("appending line");
                {
                    projectile_info += format("b %d %f\n", ownerIndex, data);
                    printl("type of double: " + typeof data);
                }
            }
        }

    }

    append_to_file("squirrel_out", projectile_info);
}