// Globals
::FLT_MAX <- 3.402823466e+38;
distances <- {};
target_bot_position <- Vector(0, 0, 0);

function Start()
{
    printl("started");

    distances <- {};
    local ent = null;
    while (ent = Entities.FindByClassname(ent, "tf_projectile_rocket"))
    {
        if (ent != null)
        {
            local owner = ent.GetOwner();
            if (owner != null)
            {
                local ownerIndex = owner.entindex();
                if (!(ownerIndex in distances)) {
                    distances[ownerIndex] <- ::FLT_MAX;
                }

                // Schedule tracking on the projectile itself, no argument needed
                EntFireByHandle(ent, "CallScriptFunction", "TrackProjectile", 0.1, null, null);
            }
        }
    }
}

function calculate_distance(pos_A, pos_B)
{
    local dx = pos_A.x - pos_B.x;
    local dy = pos_A.y - pos_B.y;
    local dz = pos_A.z - pos_B.z;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

function TrackProjectile()
{
    local projectile = self; // 'self' is the entity calling this function

    if (projectile == null)
    {
        printl("Error: projectile (self) is null");
        return;
    }

    try {
        local pos = projectile.GetOrigin();
        local owner = projectile.GetOwner();

        if (owner != null)
        {
            local ownerIndex = owner.entindex();
            local prevDistance = distances[ownerIndex];
            local currDistance = calculate_distance(pos, target_bot_position);

            if (currDistance < prevDistance)
            {
                distances[ownerIndex] <- currDistance;
            }

            // Debug output
            printl("Projectile at " + pos + " | OwnerIndex: " + ownerIndex + " | CurrDist: " + currDistance + " | MinDist: " + distances[ownerIndex]);
        }
    } catch (e) {
        printl("Error in TrackProjectile: " + e);
    }

    // Continue updating
    EntFireByHandle(projectile, "CallScriptFunction", "TrackProjectile", 0.1, null, null);
}
