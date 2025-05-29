// Globals
::FLT_MAX <- 3.402823466e+38;
::distances <- {};
::TRACKING_RATE <- 0.001

const debug = false;

IncludeScript("file_scripts");

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

function start_tracking(bot_type)
{
    ::distances <- {};
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

    while (ent = Entities.FindByClassname(ent, projectile_class))
    {
        if (ent != null)
        {
            local owner = ent.GetOwner();
            if (owner != null)
            {
                local ownerIndex = owner.entindex();
                if (!(ownerIndex in ::distances)) {
                    ::distances[ownerIndex] <- ::FLT_MAX;
                }

                EntFireByHandle(ent, "CallScriptFunction", "track_projectile", 0.1, null, null);
            }
        }
    }
}

function closest_point_on_bbox(pos, ent)
{
    local origin = ent.GetOrigin();
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

function track_projectile()
{
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

        if(target_ent == null || typeof target_ent != "instance") 
        {
            printl("target error2 ");
        }

        local target_bot_position = closest_point_on_bbox(pos, target_ent);

        if (owner != null)
        {
            local ownerIndex = owner.entindex();
            local prevDistance = ::distances[ownerIndex];
            local currDistance = calculate_distance(pos, target_bot_position);

            if (currDistance <= prevDistance)
            {
                ::distances[ownerIndex] <- currDistance;
            }
            else {
                // Finishing calculations if the distance doesn't decrease anymore
                return;
            }

            // Debug output
            if(debug) printl("Projectile at " + pos + " | OwnerIndex: " + ownerIndex + " | CurrDist: " + currDistance + " | MinDist: " + ::distances[ownerIndex]);
            
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


function send_projectile_info() 
{
    local projectile_info = "";

    if(::distances.len() == 0) 
    {
        projectile_info = "b none";
    }
    else 
    {
        foreach(ownerIndex, distance in ::distances)
        {
            projectile_info += format("b %d %f\n", ownerIndex, distance);
        }
    }

    append_to_file("squirrel_out", projectile_info);
}